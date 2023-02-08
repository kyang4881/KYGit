def tower_hanoi(n, state):
    global new_pos
    global state_pos
    state_index = [state.index(i) for i in state if n in i][0]
    initial_p = {}
    initial_p['static'] = state_index
    if n % 2 != 0:
        initial_p['disk_moved'] = ((state_index - 1) % 3)
    else:
        initial_p['disk_moved'] = ((state_index + 1) % 3)
    status = {'static': 0, 'disk_moved': 0, 'static_impossible': 0, 'disk_moved_impossible': 0}
    for key, val in initial_p.items():
        initial_pos = val
        new_pos = initial_pos
        state_pos = state_index
        call_tower(n, state, status, initial_pos, key)
    res = map_peg(status, initial_pos, initial_p)
    return res

def disk_moved():
    return new_pos != state_pos

def valid_move(n):
    is_valid = False
    if n % 2 == 0 and ((new_pos - 1) % 3 == state_pos):
        is_valid = True
    elif n % 2 != 0 and ((new_pos + 1) % 3 == state_pos):
        is_valid = True
    return is_valid

def map_peg(status, initial_pos, initial_p):
    peg_map = {0: 'A', 1: 'B', 2: 'C'}
    map_res = None
    if status['static_impossible'] + status['disk_moved_impossible'] == 2:
        peg = 'impossible'
        return peg
    else:
        if status['static_impossible'] > 0:
            peg = peg_map[initial_p['disk_moved']]
            total_count = status['disk_moved']
        else:
            peg = peg_map[initial_p['static']]
            total_count = status['static']

        return peg + " " + str(total_count)

def update_position(state_pos):
    global new_pos
    new_pos = list(set([0,1,2]) - set([new_pos, state_pos]))[0]

def call_tower(n, state, status, initial_pos, key):
    global new_pos
    global state_pos
    state_pos = [state.index(i) for i in state if n in i][0]

    if n == 1:
        if disk_moved() and valid_move(n):
            status[key] += 1
        elif not disk_moved():
            pass
        else:
            status[key + '_impossible'] += 1
    else:
        if not disk_moved():
            call_tower(n-1, state, status, initial_pos, key)
        elif disk_moved() and valid_move(n):
            status[key] += 2**(n-1)
            update_position(state_pos)
            call_tower(n-1, state, status, initial_pos, key)
        else:
            status[key + '_impossible'] += 1
