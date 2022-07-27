from functools import cache
from collections import defaultdict


def build_wall(wide: int, tall: int, bricks: list[int])->int:
    res = 0
    
    def _get_combs(curr: list[int], curr_sum: int, combs: list[tuple[int]], wide: int)->list[tuple[int]]:
        if curr_sum > wide: return
        if curr_sum == wide:
            combs.append(tuple(curr)) # for hashing later
            return
        for brick in bricks:
            _get_combs(curr+[brick], curr_sum+brick, combs, wide)
        return combs
    
    
            
    combs = _get_combs([], 0, [], wide)
    
    comb2neighbor = defaultdict(list) 
    
    seen_comb_pair = set()
    
    for i, comb in enumerate(combs):
        seen = set() # track seen edge positions
        row_sum = 0
        for brick_width in comb[:-1]:
            row_sum+=brick_width
            seen.add(row_sum)
        for j, nei in enumerate(combs):
            if (comb, nei) in seen_comb_pair or (nei, comb) in seen_comb_pair:
                comb2neighbor[comb].append(nei)
            row_sum = 0
            for brick_width in nei[:-1]:
                row_sum+=brick_width
                if row_sum in seen: break
            else: # no same edge between bottom and neighbor row
                comb2neighbor[comb].append(nei)
                seen_comb_pair.add((comb, nei))
                seen_comb_pair.add((nei, comb))
    @cache
    def dfs(comb: tuple[int], curr_tall: int)->int:
        if curr_tall == tall: return 1
        return sum(dfs(nei, curr_tall+1) for nei in comb2neighbor[comb])
    
    for comb in combs:
        res+=dfs(comb,1)
        
    l = 0
    for k in comb2neighbor:
        l+=len(comb2neighbor[k])
    return res        

def build_wall2(wide: int, tall: int, bricks: list[int]) -> int:
    """
    Returns total number of different layouts using the bricks to build a wall
    without sharing the same edge

    :param wide: wall wide in unit. e.g. 30 unit wide
    :param tall: wall tall in unit. e.g. 12 unit tall
    :param bricks: bricks are in shape of nx1. e.g. 2x1,3x1
    :return:total number of different layouts

    >>> build_wall(30, 12, [2, 3])
    16879522589829476
    >>> build_wall(30, 1, [2,3])
    1897
    """

    res = 0

    def _get_combs(
        curr: list[int], curr_sum: int, combs: list[tuple[int]], wide: int
    ) -> list[tuple[int]]:
        if curr_sum > wide:
            return
        if curr_sum == wide:
            combs.append(tuple(curr))  # for hashing later
            return
        for brick in bricks:
            _get_combs(curr + [brick], curr_sum + brick, combs, wide)
        return combs

    combs = _get_combs([], 0, [], wide)

    comb2neighbor = defaultdict(list)

    for i, comb in enumerate(combs):
        seen = set()  # track seen edge positions
        row_sum = 0
        for brick_width in comb[:-1]:
            row_sum += brick_width
            seen.add(row_sum)
        for j, nei in enumerate(combs):
            row_sum = 0
            for brick_width in nei[:-1]:
                row_sum += brick_width
                if row_sum in seen:
                    break
            else:  # no same edge between bottom and neighbor row
                comb2neighbor[comb].append(nei)

    @cache
    def dfs(comb: tuple[int], curr_tall: int) -> int:
        if curr_tall == tall:
            return 1
        return sum(dfs(nei, curr_tall + 1) for nei in comb2neighbor[comb])

    for comb in combs:
        res += dfs(comb, 1)

    return res
