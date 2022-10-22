import pygame
import math
pygame.init()

size = (width, height) = (500, 500)

screen = pygame.display.set_mode(size)
WHITE = (0,0,0)
BLUE = (11, 179, 191)
RED = (196, 24, 58)

FPS = 60

class Ball:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.xspeed = 0
        self.yspeed = 0
        self.omega = 0
        self.size = 30
        self.thetaA = 1

    def move(self,t):
        self.x += self.xspeed * t
        self.y += self.yspeed * t
        self.theta += self.omega * t * self.thetaA

    def reset(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def draw(self):
        center = (width/2,height/2)
        pos = (center[0]+self.x,center[1]+self.y)
        pygame.draw.circle(screen,BLUE,pos,self.size)

        end = (pos[0]+2*self.size*math.cos(self.theta-math.pi/2),pos[1]+2*self.size*math.sin(self.theta-math.pi/2))
        pygame.draw.line(screen,RED,pos,end)


def draw_window(ball):
    screen.fill(WHITE)
    ball.draw()
    pygame.display.update()

def main():
    run = True
    ball = Ball()
    clock = pygame.time.Clock()
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False
        
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_d]:
            ball.xspeed += 0.01
        elif pressed[pygame.K_a]:
            ball.xspeed -= 0.01
        if pressed[pygame.K_s]:
            ball.yspeed += 0.01
        if pressed[pygame.K_w]:
            ball.yspeed -= 0.01
        if pressed[pygame.K_e]:
            ball.omega += 0.001
        if pressed[pygame.K_q]:
            ball.omega -= 0.001
        if pressed[pygame.K_r]:
            ball.reset()

        ball.move(1)
        

        draw_window(ball)

        

if __name__ == "__main__":
    main()