def sol(grid):
    N = len(grid)
    is_in_grid = lambda r, c: 0 <= r < N and  0 <= c < N
    will_die = lambda r,c: grid[r][c]==1
    memo={}
    dirs=[(1,1),(0,1),(1,0)]
    def dfs(i, j): 
        if i==j==N-1:
            return 1
        if will_die(i,j):
            return 0
        if (i,j) in memo:
            return memo[(i,j)]
        p=0
        cnt=0
        new_xs, new_ys = [], []
        for dx, dy in dirs:
            new_x, new_y = i+dx, j+dy
            if is_in_grid(new_x, new_y):
                new_xs.append(new_x)
                new_ys.append(new_y)
                cnt+=1
        for new_x, new_y in zip(new_xs, new_ys):
            p+=1/len(new_xs)*(dfs(new_x, new_y))
        memo[(i,j)]=p
        return p
    return dfs(0,0)
    
sol(GRID)
            
