from cn_bidding import AgentState, Bid, award, compute_bid
from cn_scenario import Instance, Job


def _instance(jobs=(), n_agents=1, travel_time_per_unit=1.0):
    return Instance(
        n_jobs=len(jobs), n_agents=n_agents, jobs=jobs,
        agent_start_positions=tuple(0.0 for _ in range(n_agents)),
        travel_time_per_unit=travel_time_per_unit,
    )


def test_bid_finish_time_equals_free_time_plus_travel_plus_duration():
    instance = _instance(travel_time_per_unit=1.0)
    agent = AgentState(agent_id=0, position=0.0, free_time=10.0)
    job = Job(index=0, position=5.0, duration=3.0)

    bid = compute_bid(instance, agent, job)

    assert bid.travel_time == 5.0
    assert bid.finish_time == 10.0 + 5.0 + 3.0


def test_bid_scales_with_travel_time_per_unit():
    instance = _instance(travel_time_per_unit=2.0)
    agent = AgentState(agent_id=0, position=0.0, free_time=0.0)
    job = Job(index=0, position=3.0, duration=0.0)

    bid = compute_bid(instance, agent, job)

    assert bid.travel_time == 6.0


def test_award_goes_to_minimum_finish_time():
    bids = [
        Bid(agent_id=0, travel_time=0.0, finish_time=20.0),
        Bid(agent_id=1, travel_time=0.0, finish_time=15.0),
        Bid(agent_id=2, travel_time=0.0, finish_time=18.0),
    ]
    assert award(bids).agent_id == 1


def test_award_ties_broken_by_lowest_agent_id():
    bids = [
        Bid(agent_id=2, travel_time=0.0, finish_time=10.0),
        Bid(agent_id=0, travel_time=0.0, finish_time=10.0),
        Bid(agent_id=1, travel_time=0.0, finish_time=10.0),
    ]
    assert award(bids).agent_id == 0
