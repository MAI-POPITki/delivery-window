"""Cost-minimising dispatch with dated information and a fixed service deadline.

The optimiser receives present inventory, a known rail pipeline and an estimated
river window, never future actual dates. It executes one day of an LP plan and
replans when information or the observed state changes. All quantities are
divisible tonnes; vehicles, minimum lots and legal ice limits are not modelled.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class EconomicAction:
    dispatch: float
    normal: float
    alternative: float
    replanned: bool
    terminal_shortfall: float
    status: str


class EconomicPlanner:
    """Minimise costs after first minimising any unavoidable terminal shortfall.

    Freight left at the hub or in the pipeline is a committed order. Its eventual
    rail/normal-crossing charges are valued at the same rates for every policy.
    Therefore rail and normal-crossing variable costs cancel against this end
    value; alternative crossings incur their premium. This does not make rail
    free: the constant liability is included in the reported total cost.
    """

    def __init__(self, cfg, availability=None):
        self.cfg = cfg
        self.availability = None if availability is None else np.asarray(availability, dtype=bool).copy()
        if self.availability is not None and self.availability.shape != (cfg.horizon_days,):
            raise ValueError("Календарь должен покрывать весь горизонт")
        self._plan = {}
        self._token = None
        self.solves = 0

    def action(self, day, *, hub, nb, yak, backlog, in_transit,
               arrivals, river_open, reopened, close, opening, forecast_issue):
        token = (close, opening, forecast_issue, bool(reopened))
        cached = self._plan.get(day)
        state = np.array([hub, nb, yak-backlog, in_transit])
        if (cached is not None and token == self._token
                and bool(river_open) == cached['open']
                and np.allclose(state, cached['state'], rtol=0, atol=1e-5)):
            return EconomicAction(*cached['actions'], False,
                                  cached['shortfall'], cached['status'])
        self._token = token
        self._plan = self._solve(day, hub, nb, yak, backlog, in_transit,
                                 np.asarray(arrivals), river_open, reopened,
                                 close, opening)
        self.solves += 1
        today = self._plan[day]
        return EconomicAction(*today['actions'], True,
                              today['shortfall'], today['status'])

    def _solve(self, day, hub, nb, yak, backlog, in_transit,
               arrivals, river_open, reopened, close, opening):
        cfg = self.cfg
        n = cfg.horizon_days-day
        lag = cfg.rail_lead_time_days
        days = day+np.arange(n)
        is_open = (days < close) | (days >= opening)
        if reopened:
            is_open[:] = True
        elif not river_open:
            is_open = days >= max(opening, day+1)
        if self.availability is not None:
            is_open = self.availability[day:].copy()
        is_open[0] = river_open
        # Seven continuous blocks; a binary eighth block is only needed when
        # overflow is cheaper than primary storage (a non-convex tiered tariff).
        binary = (cfg.yak_overflow_capacity_tons > 0
                  and cfg.yak_overflow_cost_per_ton_day < cfg.yak_holding_cost_per_ton_day)
        size = (8 if binary else 7)*n
        x, normal, alt, stock_nb, stock_yak, pending, overflow = [i*n for i in range(7)]
        z = 7*n
        upper = np.full(size, np.inf)
        upper[x:x+n] = cfg.rail_cap_tons
        # A dispatch arriving after the deadline only changes where the same
        # committed stock sits and cannot improve the objective.
        upper[x+max(0, n-lag):x+n] = 0
        upper[normal:normal+n] = is_open*cfg.crossing_cap_normal_tons
        upper[alt:alt+n] = (~is_open)*cfg.crossing_cap_alt_tons
        upper[stock_yak:stock_yak+n] = cfg.yak_warehouse_cap_tons+cfg.yak_overflow_capacity_tons
        upper[overflow:overflow+n] = cfg.yak_overflow_capacity_tons
        if binary:
            upper[z:z+n] = 1

        def matrix(rows):
            rr, cc, vv = [], [], []
            for i, row in enumerate(rows):
                for j, value in row.items():
                    rr.append(i); cc.append(j); vv.append(value)
            return coo_matrix((vv, (rr, cc)), shape=(len(rows), size)).tocsr()

        equalities, rhs = [], []
        for i in range(n):
            row = {stock_nb+i: 1, normal+i: 1, alt+i: 1}
            if i:
                row[stock_nb+i-1] = -1
            if i >= lag:
                row[x+i-lag] = -1
            equalities.append(row)
            rhs.append((arrivals[i] if i < len(arrivals) else 0)+(nb if i == 0 else 0))
            row = {stock_yak+i: 1, pending+i: -1, normal+i: -1, alt+i: -1}
            if i:
                row[stock_yak+i-1] = -1
                row[pending+i-1] = 1
            equalities.append(row)
            rhs.append(-cfg.daily_demand_tons+(yak-backlog if i == 0 else 0))
        inequalities = [{x+i: 1 for i in range(n)}]
        limits = [hub]
        for i in range(n):
            inequalities.append({stock_yak+i: 1, overflow+i: -1})
            limits.append(cfg.yak_warehouse_cap_tons)
            if binary:
                inequalities += [
                    {overflow+i: 1, z+i: -cfg.yak_overflow_capacity_tons},
                    {stock_yak+i: -1, overflow+i: 1, z+i: cfg.yak_warehouse_cap_tons}]
                limits += [0, 0]
        aeq, beq = matrix(equalities), np.array(rhs)
        aub, bub = matrix(inequalities), np.array(limits)
        integrality = np.zeros(size, dtype=int)
        if binary:
            integrality[z:z+n] = 1

        def solve(objective, bounds_upper):
            if binary:
                result = milp(objective, integrality=integrality,
                              bounds=Bounds(np.zeros(size), bounds_upper),
                              constraints=[LinearConstraint(aeq, beq, beq),
                                           LinearConstraint(aub, -np.inf, bub)],
                              options={'mip_rel_gap': 0.})
            else:
                result = linprog(objective, A_ub=aub, b_ub=bub,
                                 A_eq=aeq, b_eq=beq,
                                 bounds=list(zip(np.zeros(size), bounds_upper)),
                                 method='highs-ds')
            return result

        objective = np.zeros(size)
        objective[alt:alt+n] = cfg.crossing_cost_alt-cfg.crossing_cost_normal
        objective[stock_nb:stock_nb+n] = cfg.nb_holding_cost_per_ton_day
        objective[stock_yak:stock_yak+n] = cfg.yak_holding_cost_per_ton_day
        objective[overflow:overflow+n] = cfg.yak_overflow_cost_per_ton_day-cfg.yak_holding_cost_per_ton_day
        objective[pending:pending+n] = cfg.shortage_penalty_per_ton_day
        terminal = pending+n-1
        upper[terminal] = 0
        result = solve(objective, upper)
        shortfall = 0.
        if not result.success and result.status == 2:
            # Do not manufacture savings by choosing to abandon a cheap order.
            # When delivery is impossible, maximise delivered tonnes first.
            upper[terminal] = np.inf
            service_objective = np.zeros(size)
            service_objective[terminal] = 1
            service_result = solve(service_objective, upper)
            if not service_result.success:
                raise RuntimeError(f'Economic service optimisation failed: {service_result.message}')
            shortfall = max(0., float(service_result.x[terminal]))
            upper[terminal] = shortfall+1e-7
            result = solve(objective, upper)
        if not result.success:
            raise RuntimeError(f'Economic cost optimisation failed: {result.message}')
        values = np.maximum(result.x, 0)
        status = 'optimal' if shortfall < 1e-6 else 'optimal_with_unavoidable_shortfall'
        plan = {}
        current_hub, current_nb = hub, nb
        current_net, current_transit = yak-backlog, in_transit
        for i in range(n):
            actions = tuple(float(values[block+i]) for block in (x, normal, alt))
            plan[day+i] = dict(actions=actions, open=bool(is_open[i]),
                state=np.array([current_hub, current_nb, current_net, current_transit]),
                shortfall=shortfall, status=status)
            sent = actions[0]
            arrived = float(arrivals[i]) if i < len(arrivals) else 0.
            if i >= lag:
                arrived += float(values[x+i-lag])
            current_hub -= sent
            current_transit += sent-arrived
            current_nb = float(values[stock_nb+i])
            current_net = float(values[stock_yak+i]-values[pending+i])
        return plan
