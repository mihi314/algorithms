import pytest

def got_cycle_recursive(graph):
    """
    Given a non-directed, connected graph, return whether it contains cycles.
    A graph is a list list of edges, each represented by a tuple of 2 "keys"/vertices.
    """
    if len(graph) == 0:
        return False

    all_neighbors = calc_neighbors(graph)

    # (randomly) choose first vertex to look at
    current = next(iter(all_neighbors.keys()))
    return search_cycle(current, None, set(), all_neighbors)

def search_cycle(current, last, visited, all_neighbors):
    if current in visited:
        return True
    visited.add(current)
    neighbors = set(all_neighbors[current])
    neighbors.discard(last) # don't immediately go "back"
    for neighbor in neighbors:
        found = search_cycle(neighbor, current, visited, all_neighbors)
        if found:
            return True
    return False


def got_cycle_iterative(graph):
    """
    Given a non-directed, connected graph, return whether it contains cycles.
    A graph is a list list of edges, each represented by a tuple of 2 "keys"/vertices.
    """
    if len(graph) == 0:
        return False

    all_neighbors = calc_neighbors(graph)

    # (randomly) choose first vertex to look at
    current = next(iter(all_neighbors.keys()))
    previous = None
    frontier = [(current, previous)]
    visited = set()
    
    while frontier:
        current, previous = frontier.pop()
        if current in visited:
            return True
        visited.add(current)

        neighbors = all_neighbors[current]
        frontier.extend((n, current) for n in neighbors if n != previous)
    return False


def calc_neighbors(graph):
    """Given a graph (list of edges, represented by a 2-tuple), return a dict {vertex: [neighbors]}"""
    neighbors = {}
    for edge in graph:
        assert len(edge) == 2
        neighbors.setdefault(edge[0], []).append(edge[1])
        neighbors.setdefault(edge[1], []).append(edge[0])
    return neighbors


@pytest.fixture(params=[got_cycle_recursive, got_cycle_iterative])
def got_cycle(request):
    return request.param

def test_trivial(got_cycle):
    assert(got_cycle([]) == False)
    assert(got_cycle([(1, 2)]) == False)
    assert(got_cycle([(1, 1)]) == True)

def test_loop(got_cycle):
    graph = [(1, 2), (3, 2), (3, 1)]
    assert(got_cycle(graph) == True)

    graph = [(1, 2), (2, 3), (3, 1)]
    assert(got_cycle(graph) == True)

def test_more_complicated(got_cycle):
    graph = [(1, 2), (2, 3), (3, 1), (3, 4), (6, 3)]
    assert(got_cycle(graph) == True)

    graph = [(1, 2), (2, 3), (3, 6), (3, 4), (6, 4)]
    assert(got_cycle(graph) == True)

    graph = [(1, 2), (2, 3), (3, 6), (3, 4), (2, 5)]
    assert(got_cycle(graph) == False)


if __name__ == "__main__":
    got_cycle_iterative([(1, 2)])
