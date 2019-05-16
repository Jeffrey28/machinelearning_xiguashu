#### 动态规划求解MDPs的Planning

强化学习的目的就是求解MDP的最优策略，使其在任意初始状态下，都能获得最大的$v_\pi$值。

动态规划是一种通过把复杂问题划分为子问题，并对自身问题进行求解，最后把子问题的解结合起来解决原问题的方法。[动态]是指问题由一系列的动态组成，而且状态能一步步地改变，[规划]即优化每一个子问题。因为MDP的马尔可夫性，即某一时刻的子问题仅仅取决于上一时刻的子问题的action，并且Bellman方程可以递归地切分子问题，所以我们可以采用动态规划来求解Bellman方程。

MDP的问题主要分两类

- Prediction问题
  - 输入：MDP $ < S,A,P,R,\gamma  > $和策略（policy）$\pi $
  - 输出：状态价值函数${v_\pi }$

- Control问题
  - 输入：MDP $ < S,A,P,R,\gamma  > $
  - 输出：最优状态价值函数${v_*}$和最优策略$\pi_*$

回顾上一章提到的状态值函数和行为值函数的贝尔曼最优方程：
$$
{v_*}(s) = \mathop {\max }\limits_a E[{R_{t + 1}} + \gamma {v_*}({S_{t + 1}})|{S_t} = s,{A_t} = a]\\= \mathop {\max }\limits_a \sum\limits_{s',r} {p(s',r|s,a)[r + \gamma } {v_*}(s')]
$$

$$
{q_*}(s,a) = E[{R_{t + 1}} + \gamma \mathop {\max }\limits_{a'} {q_*}({S_{t + 1}},a')|{S_t} = s,{A_t} = a] \\= \sum\limits_{s',r} {p(s',r|s,a)[r + \gamma } \mathop {\max }\limits_{a'} {q_*}(s',a')]
$$

##### 策略评估（Policy Evaluation)

将所有一步状态转换的可能性都进行评估的操作称作**完全备份**。因为DP算法完全掌握环境模型，所以知道给定策略下的状态转换概率，可以使用完全备份的方法迭代更新状态价值函数。

首先对于任意的策略$\pi$，我们如何计算其状态值函数$v_\pi(s)$? 这个问题被称作$\color{red}策略估计$,

对于确定性策略，值函数为
$$
{v_\pi }(s) = {E_\pi }[{G_t}|{S_t} = s] \\= {E_\pi }[{R_{t + 1}} + \gamma {G_{t + 1}}|{S_t} = s] \\= {E_\pi }[{R_{t + 1}} + \gamma {v_\pi }({S_{t + 1}})|{S_t} = s] \\= \sum\limits_a {\pi (a|s)\sum\limits_{s',r} {p(s',r|s,a)[r + \gamma {v_\pi }} } (s')]
$$
  上式表示在某策略$\pi$下，$\pi(s)$有多种可能时的状态值函数

状态$s$处的值函数$v_\pi(s)$, 可以利用后继状态$v_\pi(s')$来表示。但是$v_\pi(s')$也是未知的，那么怎么计算当前状态的值函数，这不是自己抬自己吗？是的，正是bootstrapping算法（自举算法）

方程（3）中唯一未知数是值函数，所以其是关于值函数的线性方程组，仅用一个数组保存各状态值函数，每当得到一个新值，就将旧的值覆盖，较好的利用新值，收敛快。
$$
{v_{k + 1}}(s) = {E_\pi }[{R_{t + 1}} + \gamma {v_\pi }({S_{t + 1}})|{S_t} = s] \\= \sum\limits_a {\pi (a|s)\sum\limits_{s',r} {p(s',r|s,a)[r + \gamma {v_k}} } (s')]
$$
![img](201019414696.png)

​    **Example:**

![1539615185520](1539615185520-1539621307611.png)


$$
{v_2}(1) = \pi (up|1)p(1, - 1|1,up)[ - 1 + {v_1}(1)] \\+ \pi (down|1)p(5, - 1|1,down)[ - 1 + {v_1}(5)] \\+ \pi (right|1)p(2, - 1|1,right)[ - 1 + {v_1}(2)] \\+ \pi (left|1)p(shaded, - 1|1,left)[ - 1 + 0]=-1.75
$$

##### 策略改进（Policy Improvement)

进行策略估计的目的，是为了寻找更好的策略，这个过程叫做$\color{red}策略改进$（Policy Improvement)。

假设我们有一个策略$\pi$，并且确定了它的所有状态的值函数$v_\pi(s)$。如果我们在状态$s$下采用动作$a \ne \pi (s)$，是否会更好。判断好坏，需要计算行为值函数${q_\pi }(s,a)$
$$
{q_\pi }(s,a) = E[{R_{t + 1}} + \gamma {v_\pi }({S_{t + 1}})|{S_t} = s,{A_t} = a] \\= \sum\limits_{s',r} {p(s',r|s,a)[r + \gamma {v_\pi }} (s')]
$$
$\color{red}评价标准$：${q_\pi }(s,a)$是否大于$v_\pi(s)$. 如果${q_\pi }(s,a)>v_\pi(s)$，那么至少说明新策略[仅在状态$s$下采用动作$a$，其他状态下遵循策略$\pi$]比旧策略[所有状态下都遵循策略$\pi$]整体上要更好。

$\color{red}策略改进定理$（policy improvement theorem）:$\pi$和$\pi'$是两个确定的策略，如果对所有状态$s$有${q_\pi }(s,\pi '(s)) \geqslant {v_\pi }(s)$，那么策略$\pi'$必然比策略$\pi$更好，或者至少一样好。
上式子等价于${v_{\pi '}}(s) \geqslant {v_\pi }(s)$。

有了在某状态$s$上改进策略的方法和策略改进定理，我们可以遍历所有状态和所有可能的动作$a$，并采用贪心策略来获得新策略$\pi'$。即对所有的$s \in S$，采用下式更新策略：
$$
\pi '(s) = \mathop {\arg \max }\limits_a {q_\pi }(s,a) = \mathop {\arg \max }\limits_a E[{R_{t + 1}} + \gamma {v_\pi }({S_{t + 1}})|{S_t} = s,{A_t} = a] \\= \mathop {\arg \max }\limits_a \sum\limits_{s',r} {p(s',r|s,a)[r + \gamma {v_\pi }(s')]}
$$
这种采用关于值函数的贪心策略获得新策略，改进旧策略的过程，称为策略改进（policy improvement）

贪心策略能否收敛到最优策略?

假设策略改进过程已经收敛，即${v_{\pi '}} \geqslant {v_\pi }$。根据上面的策略更新的式子，可以知道对于所有的$s \in S$下式成立：
$$
{v_{\pi '}}(s) = \mathop {\max }\limits_a E[{R_{t + 1}} + \gamma {v_{\pi '}}({S_{t + 1}})|{S_t} = s,{A_t} = a] = \mathop {\max }\limits_a \sum\limits_{s',r} {p(s',r|s,a)[r + \gamma {v_{\pi '}}(s')]}
$$
此式子正好就是Bellman optimality equation，因此${v_{\pi '}}$肯定是$v_*$。且${v_{\pi '}}$与$ {v_\pi }$都是最优策略。

##### 策略迭代（Policy Iteration)

略迭代算法就是上面两块的结合。假设我们有一个策略$\pi$，那么可以用policy evaluation获得它的值函数$v_\pi(s)$，然后根据policy improvement得到更好的策略$\pi'$，接着再计算$v_{\pi‘}(s)$，再获得更好的策略$\pi''$，整个过程顺序进行如：![img](201019436259.png)

在每次评估得到值函数后，针对每个$v(s)$，从里面选择一个action，使得下一时刻获得的回报最大，对选择出来出来的action，将其概率置1，其余action置0即可。贪婪策略是指每次选择的策略都是基于当前最优的策略。理论已经证明采用这种简单策略可以保证收敛到最优策略。

![1539530452058](1539530452058.png)

##### 价值迭代（Value Iteration)

进行策略改善之前一定要等到策略值函数收敛吗？

策略迭代中每一次策略评估也是一个迭代过程，需要等到其收敛后才能在进行策略改进。

从上面的图可以看出$k=10$与$k=\infty$所得到的贪婪策略是一样的。

策略改进过程是可以简化的。如果在评估一次之后就进行策略改善，则称为值函数迭代算法。进行一次扫描（对每个状态备份一次）后就停止策略评价。可以表述为将策略改进和简化的策略评价相结合的一个简单的备份过程。

$v_{k+1}(s) = \underset{a}{max}\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]$，one-step lookahead对应：$\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]$

![1539616800189](1539616800189-1539621307612.png)



# 强化学习基础实验（1）——Gridworld

# Gridworld

```
通过这个实验，了解MDP的Dynamic Programming解法
```

所有的实验源代码都在`lib`目录下，来自[dennybritz](https://github.com/dennybritz/reinforcement-learning)，这里只做解读和归总。

## 实验目录

- [Gridworld](https://applenob.github.io/gridworld.html)：对应MDP的Dynamic Programming
- [Blackjack](https://applenob.github.io/black_jack.html)：对应Model Free的Monte Carlo的Planning和Controlling
- [Windy Gridworld](https://applenob.github.io/windy_gridworld.html)：对应Model Free的Temporal Difference的On-Policy Controlling，SARSA。
- [Cliff Walking](https://applenob.github.io/cliff_walking.html)：对应Model Free的Temporal Difference的Off-Policy Controlling，Q-learning。
- [Mountain Car](https://applenob.github.io/mountain_car.html)：对应Q-Learning with Linear Function Approximation。
- [Atari](https://applenob.github.io/atari.html)：对应Deep-Q Learning。

## 本文目录

- [问题介绍](https://applenob.github.io/gridworld.html#%E9%97%AE%E9%A2%98%E4%BB%8B%E7%BB%8D)
- [Policy Evaluation](https://applenob.github.io/gridworld.html#Policy-Evaluation)
- [Policy Iteration](https://applenob.github.io/gridworld.html#Policy-Iteration)
- [Value Iteration](https://applenob.github.io/gridworld.html#Value-Iteration)

## 问题介绍

![img](grid_world.png)

你的agent在一个M×N的格子世界里，想办法走到左上角或者右下角的终点。

这个实验用来解释如何在MDP的环境里，使用动态规划（DP），来寻找最佳策略$\pi_*$。方法是：`Policy Iteration`和`Value Iteration`。

为什么说这个实验是MDP呢？回顾下MDP的五元组：$<\textbf{S}, \textbf{A}, \textbf{P}, \textbf{R}, \gamma>$，分别对应：

- $\textbf{S}$：状态集合，上图一共有16种状态，对应16个格子。
- $\textbf{A}$：动作集合，上下左右四种。
- $\textbf{P}$：状态转移概率矩阵，即当前状态是$s$且动作是$a$时，下一个状态是$s+1$的概率。这里如果动作选定，下一个状态是唯一的。
- $\textbf{R}$：奖励函数，即在某一个状态$s$，下一时刻能收获的即时奖励的期望。这里假设除了最终状态，每到另一个状态，即时奖励都是-1。
- $\gamma$：折扣系数，题目没有明确给出。

来总结一下：五元组中，$\textbf{S}$和$\textbf{R}$用来描述环境，即环境有什么样的状态，不同的状态下有什么回报。$\textbf{A}$描述agent的可操作动作范围。$\textbf{P}$最重要，它描述agent和环境如何交互，即，在某状态下采取某个动作会如何进到下一个状态；并且$\textbf{P}$蕴含着MDP的思想，即`在某个时刻，agent所处在的状态，只和它上一个时刻所处的状态相关。`



```python
import numpy as np
import sys
from gym.envs.toy_text import discrete

# 定义动作
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
```



- state_num = m × n
- P : [state_num, action_num, 4]，其中最后一维的4位分别表示： `prob`, `next_state`, `reward`, `is_done`。

In [1]:

```python
import numpy as np
from lib.envs.gridworld import GridworldEnv
```

In [2]:

```python
env = GridworldEnv()
```



## Policy Evaluation

![img](iterative_policy_evaluation.png)

In [3]:

```python
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)
```



- $v_{\pi}(s) = \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')]\;\;\forall s \in S$
- V[s] = $\sum$ action_prob × prob × (reward + discount_factor × V[next_state])
- 代码和公式正好一一对应。

In [4]:

```python
random_policy = np.ones([env.nS, env.nA]) / env.nA
```



- policy: [state_num, action_num]
- 测试期望的value和评估的value误差小于一定的范围。
- `abs(desired-actual) < 1.5 * 10**(-decimal)`

In [5]:

```python
v = policy_eval(random_policy, env)
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
```



代码通过测试。



## Policy Iteration

![img](policy_iteration.png)

In [9]:

```python
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            # 取evaluation后的value最大的action
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V
```

In [13]:

```python
policy, v = policy_improvement(env)
actions = ["UP", "RIGHT", "DOWN", "LEFT"]
print([[action for num, action in zip(one, actions) if num == 1.0] for one in policy])
```



```python
[['UP'], ['LEFT'], ['LEFT'], ['DOWN'], ['UP'], ['UP'], ['UP'], ['DOWN'], ['UP'], ['UP'], ['RIGHT'], ['DOWN'], ['UP'], ['RIGHT'], ['RIGHT'], ['UP']]
```



![img](policy_evaluation_gw1.png)

![img](policy_evaluation_gw2.png)

可以看到输出的policy已经收敛。



## Value Iteration

![img](value_iteration.png)

In [15]:

```python
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V
```



整体思路：每次通过one-step lookahead更新v(s)，找到让v(s)最大的action。直到找到最终的$v_*(s)$，每次选取的动作可以构成一个deterministic policy。

更新公式：$v_{k+1}(s) = \underset{a}{max}\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]$，one-step lookahead对应：$\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]$

In [16]:

```python
policy, v = value_iteration(env)
print([[action for num, action in zip(one, actions) if num == 1.0] for one in policy])
```



```python
[['UP'], ['LEFT'], ['LEFT'], ['DOWN'], ['UP'], ['UP'], ['UP'], ['DOWN'], ['UP'], ['UP'], ['RIGHT'], ['DOWN'], ['UP'], ['RIGHT'], ['RIGHT'], ['UP']]
```



# 策略迭代和值迭代之间的区别

# 前言

在基于模型的强化学习中，我们有两种方法可以对策略进行求解，那就是策略迭代和值迭代。这两个方法有些类似，初学时在这里犯了迷糊。在看了几篇博客的解释后，目前对两者之间的区别也有了一些认识，因此写下这篇博客，做为记录。该笔记更偏向于算法形式上的区别，帮助初学者理解算法的各个步骤。

# 一个例子

在介绍两者之间的区别时，首先举一个例子，这个例子用于说明策略迭代和值迭代两种方法中，值v(s)更新方式的不同。这个例子如下图，是一个状态转换图。

![img](1-1539621307611.png)

​                                                                                    状态转换图例子

这个图很简单，是一个马尔科夫决策过程图，包含四个状态，两个动作，奖励，转移概率等信息。在这个例子中，有一个需要注意的地方：在状态s0,执行动作a1后，其后继状态有两个。所以，在某个状态下，执行某个动作，其后继状态不一定是确定的，也可能是像执行动作a1一样，有一定的概率转移到不同的状态(在这里是s1和s2)。关于这个问题，我最开始陷入了一个误区，认为在某个状态下执行某个动作，其后继状态是一定的，这种想法影响了后面对值函数更新的理解。而在后来想出这个例子后，对值函数更新的理解更加清晰了。再举一个例子来解释这种执行一个动作后有多个后继状态的现象：你现在在A地，需要赶往B地，可供你选择的方式有{走路，乘公交，骑车}（可以当作动作集），你选择了走了前往。但是，在前往的路上，有一定的机率会发生交通事故，如果正好不幸，在你身上发生了交通事故，那么你的下一个状态不是在B地，而是在医院；如果路上什么也没有发生，则顺利到达B地，所以下一个状态就是在B地。这样，应该对一个动作有多个后继状态这种情况有比较清楚的认识了。这个例子同时也解释了Richard S. Sutton .etc的《Reinforcement Learning: An Introduction》一书中的类似下面的图（空心圆表示状态，实心黑色原点表示动作，动作a后跟着两个后继状态）：

# 策略迭代和值迭代

在这部分，简单对伪代码进行说明，了解策略迭代和值迭代两种算法的大致流程。然后，我们通过上面提到的那个例子，来手工计算一下某一个时刻状态值的更新，通过这样一个具体的计算过程，可以更加清楚的看到两者在值函数更新方面的不同。

![img](2-1539621307612.png)

策略迭代和值迭代伪代码

## 策略迭代步骤

1. 初始化（Initialization）：初始化所有的v(s)和π(s)
2. 策略评估（Policy Evaluation）：在当前的策略π下，计算每一个状态的v(s)，直到v(s)收敛。状态值的更新在后面用上面提到的例子进行说明。
3. 策略提升（Policy Improvement）：对于每一个状态，尝试使用2中计算出来的状态值来进行更新。如果所有的状态的策略都没有改变，说明当前策略已经稳定，算法结束。只要有一个策略发生了改变，则说明算法还没有稳定，在更新策略后，回到第2步，在新的策略条件下，重新计算值函数。重复2、3步骤，直到v(s)heπ(s)都收敛，算法结束。

## 值迭代步骤

1. 初始化所有的v(s)
2. 对于所有的状态，根据贝尔曼方程对状态值进行更新。但是需要注意的是，伪代码中有个max，也就是说，对于当前状态s可以执行的每个动作，都计算一下执行这个动作并到达下一个状态的期望值，然后取期望最大的价值作为当前状态s的状态值v(s)。循环的执行这个步骤，最后值函数会收敛，我们就可以得到最有值函数了。
3. 根据第2步收敛的状态值函数，计算出每隔状态下应该采取的最有动作，就是最后的策略了。

## 策略迭代和值迭代的区别

在认真看伪代码和上面对伪代码的文字说明后，应该已经可以感觉到两者之间的不同了。下面分别说明每个步骤上两者之间的区别。

1. 初始化方面：

策略迭代需要对策略进行初始化，而值迭代不需要。

1. 在值函数的更新上：

首先，策略迭代也是需要计算值函数的，策略的确定需要依靠值函数。但与值迭代不同的是，策略迭代的值函数计算是在当前策略π的指导下进行，而值迭代的值函数更新完全不涉及策略π。其次，值迭代更新值函数时，需要计算当前状态下可能执行的每一个动作的期望值，而策略迭代只需要计算当前策略π确定的当前状态下应该执行的动作的期望值。这里状态值的更新比较重要，也是“一个例子”中举例的目的。下面，我们根据最开始给出的那个状态转换图的例子，来手工计算一遍（对初学者友好）。我们假设，目前正处于状态s0，需要更新v(s0)，折扣因子gamma=0.9:

对于策略迭代，在这个例子中有两种情况：

（1）如果策略π(s0)=a1，则值函数更新的计算过程为：
$$
v(s_0)=\sum_{s^{'}}p(s^{'}|s,a_1)[r(s,a_1,s^{'})+\gamma v(s^{'})]
=0.7*(-1+0.9*3)+0.3*(-1+0.9*4)
=1.97
$$
（2）如果策略π(s0)=a2，则值函数更新计算过程为：
$$
\begin{equation}
\begin{split}
v(s_0)&=&\sum_{s^{'}}p(s^{'}|s,a_2)[r(s,a_2,s^{'})+\gamma v(s^{'})]\\
&=&1*(-2+0.9*5)\\
&=&2.5
\end{split}
\end{equation}
$$
对于值迭代，s0的值函数更新计算过程为：
$$
\begin{equation}
\begin{split}
v(s_0) &=& max(\sum_{s^{'}}p(s^{'}|s,a_1)[r(s,a_1,s^{'})+\gamma v(s^{'})],\sum_{s^{'}}p(s^{'}|s,a_2)[r(s,a_2,s^{'})+\gamma v(s^{'})])\\
&=&max(0.7*(-1+0.9*3)+0.3*(-1+0.9*4),1*(-2+0.9*5))\\
&=&max(1.97,2.5)\\
&=&2.5
\end{split}
\end{equation}
$$


1. 在策略更新或者策略确定上：
   这方面，两者计算方法都是一样的，都是根据上一步计算好的状态值，求能够取得最大期望值的那个动作，作为在某个状态下应该执行的动作，也就是确定了状态到动作的映射。不同的是在这一步里，策略迭代可能还没有确定最终的策略，它在这步更新策略后可能会发现策略的不稳定，于是需要重新进行策略评估，然后策略提升，如此反复迭代。而值迭代在这一步会立即得到一个确定的策略，不需要再回过头计算值函数。

## 策略迭代和值迭代更本质上的区别

上面提到的区别都是比较表面的，包括上面通过例子来手工计算值函数的跟新，都是为了说明两者在形式上的差别。需要了解更加深入两者之间的区别，以及算法为什么这样设计，两者在收敛情况上的差别，可以看看参考资料[2]中知乎上的讨论，也许会有所帮助。

# 参考资料

[1]<http://blog.csdn.net/panglinzhuo/article/details/77752574>

[2]<https://www.zhihu.com/question/41477987>