import pygame, sys
from pygame.locals import *
from math import sin, cos, radians
from backend import *

pygame.init()

SZ_WINDOW = 800
TIMETICK = 20
SZ_BOB = 15
epsilon_theta = [3, 2, 5]
epsilon_omega = [2, 2, 4]
epsilon_curr = [2, 4, 8, 6, 10, 12]

window = pygame.display.set_mode((SZ_WINDOW, SZ_WINDOW))
pygame.display.set_caption("Inverted Pendulum Fuzzy Logic")

pygame_screen = pygame.display.get_surface()
pygame_screen.fill((255, 255, 255))

PIVOT = (SZ_WINDOW //2 , 9 * SZ_WINDOW // 10)
LEN_SWING = 320


class Mass(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.theta = -1
        self.dtheta = 1
        self.rect = pygame.Rect(int(PIVOT[0] - LEN_SWING * cos(self.theta)),
                                int(PIVOT[1] - LEN_SWING * sin(self.theta)),
                                1, 1)
        self.draw()

    def compute_angle(self):

        current = compute_current(
            self.theta, self.dtheta, epsilon_theta, epsilon_omega, epsilon_curr)
        theta_new = self.theta + self.dtheta / 10 + current / 200
        omega_new = self.dtheta + current / 10
        self.theta, self.dtheta = theta_new, omega_new
        print(theta_new, omega_new)

        self.rect = pygame.Rect(PIVOT[0] -
                                LEN_SWING * sin(self.theta),
                                PIVOT[1] -
                                LEN_SWING * cos(self.theta), 1, 1)

    def draw(self):
        pygame.draw.circle(pygame_screen, (0, 0, 0), PIVOT, 5, 0)
        pygame.draw.circle(pygame_screen, (0, 0, 0), self.rect.center, SZ_BOB, 0)
        pygame.draw.aaline(pygame_screen, (0, 0, 0), PIVOT, self.rect.center)
        pygame.draw.line(pygame_screen, (0, 0, 0), (0, PIVOT[1]), (SZ_WINDOW, PIVOT[1]))

    def update_pygame_screen(self):
        self.compute_angle() # recompute the angle.
        pygame_screen.fill((255, 255, 255))
        self.draw()


bob = Mass()
clock = pygame.time.Clock()


TICK = USEREVENT # set the time tick as the user event of the pygame.
pygame.time.set_timer(TICK, TIMETICK)


def input(events):
    for event in events:
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == TICK:
            bob.update_pygame_screen()

while True:
    input(pygame.event.get())
    pygame.display.flip()