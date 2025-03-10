import numpy as np
import cv2


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def ToNumpy(self):
        return np.array([self.x, self.y], np.float32)

    def distance(self, target):
        return np.sqrt((self.x - target.x)**2 + (self.y - target.y)**2)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

    
class States:
    
    def __init__(self,x,y,targetPosition, x_limit = 100, y_limit = 100):
        self.position = Position(x,y)
        self.targetPosition = targetPosition
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.distanceWithTarget = (self.position.distance(targetPosition))

    def ToNumpy(self):
        return np.array([self.position.x/self.x_limit/2,
                         self.position.y/self.y_limit/2,
                         self.targetPosition.x/self.x_limit/2,
                         self.targetPosition.y/self.y_limit/2,
                         self.distanceWithTarget], np.float32)

class NAV:
    def __init__(self):
        self.x_limit = 100
        self.y_limit = 100
        self.visited_positions = set()
        self.map = np.zeros((self.x_limit, self.y_limit), np.int32)
        self.target = Position(99, 99)
        self.cur_step = 0
        self.targetPlanA = Position(0, 0)
        self.targetPlanB = Position(0, self.y_limit-1)
        self.targetPlanC = Position(self.x_limit-1, 0)
        self.targetPlanD = Position(self.x_limit-1, self.y_limit-1)

        self.map[self.target.x][self.target.y] = 1
        self.countOfReachingTarget = 0
        self.state = States(50, 50, self.target, self.x_limit, self.y_limit)
        self.obstacle = [(70, 10), (80, 20), (25, 70), (20, 20)]

        self.action_space = (4)
        self.observation_space = (5)
    def step(self, action):
        reward = -1
        self.cur_step += 1
        done = False
        self.render()
        if   action == 0:
            self.state.position.x += 1
        elif action == 1:
            self.state.position.x -= 1
        elif action == 2:
            self.state.position.y += 1
        elif action == 3:
            self.state.position.y -= 1
        elif action == 4:
            self.state.position.x -= 1
            self.state.position.y -= 1
        elif action == 5:
            self.state.position.x += 1
            self.state.position.y -= 1
        elif action == 6:
            self.state.position.x -= 1
            self.state.position.y -= 1
        elif action == 7:
            self.state.position.x += 1
            self.state.position.y -= 1

        distanceWithTarget = (self.state.position.distance(self.state.targetPosition))
        reward += (self.state.distanceWithTarget - distanceWithTarget) * 10  # 獎勵隨距離變化

        if self.state.position in self.visited_positions:
            if self.state.distanceWithTarget > distanceWithTarget:
                reward += 1
            reward -= 2  # 回頭路額外懲罰
        else:
            self.visited_positions.add(self.state.position)  # 記錄走過的路
        self.state.distanceWithTarget = distanceWithTarget

        if self.state.position.x >= self.x_limit or self.state.position.x < 0 or self.state.position.y >= self.y_limit or self.state.position.y < 0:
            done = True
            reward = -200
        if self.target.distance(self.state.position) < 5:
            done = True
            self.countOfReachingTarget += 1
            reward += 100
        
        return self.state.ToNumpy(), reward, done, False, {}
    
    def reset(self):
        # if self.cur_step < 10000:
        #     self.target  = self.targetPlanA
        # elif self.cur_step < 20000:
        #     self.target  = self.targetPlanB
        # elif self.cur_step < 30000:
        #     self.target  = self.targetPlanC
        # else:
        #     self.target  = self.targetPlanD
        if np.random.rand() < 0.25:
            self.target  = self.targetPlanA
        elif np.random.rand() < 0.50:
            self.target  = self.targetPlanB
        elif np.random.rand() < 0.75:
            self.target  = self.targetPlanC
        else:
            self.target  = self.targetPlanD

        self.visited_positions.clear()
        self.state = States(50, 50, self.target)
        return self.state.ToNumpy(), {}
    
    def render(self):

        img = np.zeros((self.x_limit, self.y_limit, 3), np.uint8)
        img[self.state.position.x][self.state.position.y] = [0, 255, 0]
        img[self.target.x][self.target.y] = [0, 0, 255]
            
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("NAV", img)
        cv2.waitKey(10)

if __name__ == "__main__":
    env = NAV()
    state = env.reset()
    done = False

    while not done:

        action = np.random.choice([0, 1, 2, 3])
        next_state, reward, done, _, _ = env.step(action)
        env.render()

        state = next_state

    cv2.destroyAllWindows()
