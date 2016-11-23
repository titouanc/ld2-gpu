/* Return the index of cell i,j in the nth world,
   where worlds have a width w and a height h */

#define  WIDTH get_global_size(2)
#define HEIGHT get_global_size(1)
#define WORLDS get_global_size(0)

#define THIS_WORLD get_global_id(0)

#define idx(i, j) THIS_WORLD * WIDTH * HEIGHT +\
                  ((i) % HEIGHT) * WIDTH +\
                  ((j) % WIDTH)

                                        // Def   Col  
__constant float PAYOFFS[2][2] = {{${P}, ${T}},  // Defect (me)
                                  {${S}, ${R}}}; // Collaborate (me)

#define payoff(me, other) PAYOFFS[me][other]

__kernel void play(__global const int *players,
                   __global float *rewards)
{
    int i = get_global_id(1);
    int j = get_global_id(2);
    
    bool me = players[idx(i, j)];
    int res = 0;

    /* Von Neumann */
    res += payoff(me, players[idx(i-1,   j)]);
    res += payoff(me, players[idx(  i, j-1)]);
    res += payoff(me, players[idx(  i, j+1)]);
    res += payoff(me, players[idx(i+1,   j)]);
    
    // Mako templating for lattice type
    % if moore:
        res += payoff(me, players[idx(i+1, j-1)]);
        res += payoff(me, players[idx(i-1, j-1)]);
        res += payoff(me, players[idx(i+1, j+1)]);
        res += payoff(me, players[idx(i-1, j+1)]);
    % endif

    rewards[idx(i, j)] = res;
}


__kernel void choose_best(__global const int *players,
                          __global const float *rewards,
                          __global int *after)
{
    int i = get_global_id(1);
    int j = get_global_id(2);
    
    int his_idx, my_idx = idx(i, j);
    float score, best_score = rewards[idx(i, j)];
    bool best_strat = players[my_idx];

    /* Von Neumann: up/down and left/right */
    for (int d=-1; d<=1; d+=2){
        his_idx = idx((i+d), j);
        score = rewards[his_idx];
        if (score >= best_score) {
            best_strat = players[his_idx];
            best_score = score;
        }

        his_idx = idx(i, (j+d));
        score = rewards[his_idx];
        if (score >= best_score) {
            best_strat = players[his_idx];
            best_score = score;
        }
    }

    // Mako templating for lattice type
    % if moore:
        for (int di=-1; di<=1; di+=2){
            for (int dj=-1; dj<=1; dj+=2){
                his_idx = idx((i+di), (j+dj));
                score = rewards[his_idx];

                if (score > best_score) {
                    best_strat = players[his_idx];
                    best_score = score;
                }
            }
        }
    % endif
    
    after[idx(i, j)] = best_strat;
}


__kernel void choose_replicator(__global const float *probas,
                                __global const float *choice,
                                __global const   int *players,
                                __global const float *rewards,
                                __global int *after)
{
    int i = get_global_id(1);
    int j = get_global_id(2);

    int my_idx = idx(i, j);

    // Choose random neighbor
    % if moore:
        int k = 8;
        int neighbors[8] = {
            idx(i-1, j-1), idx(i-1, j), idx(i-1, j+1),
            idx(  i, j-1),              idx(  i, j+1),
            idx(i+1, j-1), idx(i+1, j), idx(i+1, j+1)
        };
    % else:
        int k = 4;
        int neighbors[4] = {
                         idx(i-1, j),
            idx(i, j-1),              idx(i, j+1),
                         idx(i+1, j)
        };
    % endif

    unsigned int my_choice = k * choice[my_idx];
    unsigned int his_idx = neighbors[my_choice];
    
    float Pij = 1 + (rewards[his_idx] - rewards[my_idx]) / (k * ${DPMax});
    Pij /= 2;

    float p = probas[my_idx];
    after[my_idx] = (p < Pij) ? players[his_idx] : players[my_idx];
}


__kernel void sum_world(__global const int *players,
                        __global int *coop_lvl,
                        const int base, const int w, const int h)
{
    int N = get_global_id(0);

    int idxmin = N*w*h;
    int idxmax = (N+1)*w*h;
    int res = 0;

    for (int i=idxmin; i<idxmax; i++){
        res += players[i];
    }
    
    coop_lvl[base + N] = res;
}
