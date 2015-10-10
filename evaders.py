import pygame
from pygame.locals import *
import random
import sys

pygame.init()

WIDTH = 640
HEIGHT = 480

RIGHT_BOUND = 590
LEFT_BOUND = 10

SPACING = 50

MAX_ENEMY_MISSILE = 3

enemyspeed = 3

enemies = []
ourmissiles = []
enemymissiles = []

screen = pygame.display.set_mode((WIDTH,HEIGHT))
backdrop = pygame.image.load('data/background.bmp')

pygame.key.set_repeat(1, 1)
pygame.display.set_caption('Evaders')


class Sprite:
	def __init__(self, xpos, ypos, filename):
		self.image = pygame.image.load(filename)
		# self.image.set_colorkey((0,0,0))
		self.rect = self.image.get_rect()
		self.rect.center = (xpos, ypos)
	def set_position(self, xpos, ypos):
		self.rect.center = (xpos, ypos)
	def render(self):
		screen.blit(self.image, self.rect)

def reset_game():
	global ourmissiles
	ourmissiles = []
	global enemymissiles
	enemymissiles = []
	global enemies
	enemies = []
	x = 0
	for count in range(10):
		enemies.append(Sprite(SPACING * x + SPACING, SPACING, 'data/alien.png'))
		enemies.append(Sprite(SPACING * x + SPACING, SPACING*2, 'data/alien.png'))
		enemies.append(Sprite(SPACING * x + SPACING, SPACING*3, 'data/alien.png'))
		x += 1
	hero.set_position(WIDTH/2, 400)
	pygame.time.delay(1000)

def updateEnemies():
	global enemyspeed
	for count in range(len(enemies)):
		enemies[count].rect.x += enemyspeed
		enemies[count].render()

	if enemies[len(enemies)-1].rect.x > RIGHT_BOUND:
		enemyspeed = -3
		for count in range(len(enemies)):
			enemies[count].rect.y += 5

	if enemies[0].rect.x < LEFT_BOUND:
		enemyspeed = 3
		for count in range(len(enemies)):
			enemies[count].rect.y += 5

def updateMissiles():
	for missile in ourmissiles:
		if missile.rect.y < HEIGHT and missile.rect.y > 0:
			missile.render()
			missile.rect.y -= 5
		else:
			ourmissiles.remove(missile)
	for missile in enemymissiles:
		if missile.rect.y < HEIGHT and  missile.rect.y > 0:
			missile.render()
			missile.rect.y += 3
		else:
			enemymissiles.remove(missile)

def checkCollisions():
	for e in range(len(enemymissiles)):
		if hero.rect.colliderect(enemymissiles[e].rect):
			reset_game()
			return

	for count in range(0, len(enemies)):
		if pygame.sprite.collide_rect(hero, enemies[count]):
			reset_game()
			return
		for e in range(len(ourmissiles)):
			if pygame.sprite.collide_rect(ourmissiles[e], enemies[count]):
				del enemies[count]
				del ourmissiles[e]
				return

def fire():
	missile = Sprite(0, HEIGHT, 'data/hero_missile.png')
	missile.set_position(hero.rect.centerx, hero.rect.centery)
	ourmissiles.append(missile)

def bomb():
	if len(enemymissiles) < MAX_ENEMY_MISSILE:
		missile = Sprite(0, HEIGHT, 'data/alien_missile.png')
		missile.set_position(enemies[random.randint(0, len(enemies) - 1)].rect.centerx, enemies[0].rect.centery)
		enemymissiles.append(missile)

# Init game
hero = Sprite(20, 400, 'data/hero.png')
reset_game()

while True:
	screen.blit(backdrop, (0, 0))

	# Key event handling
	for ourevent in pygame.event.get():
		if ourevent.type == QUIT:
			pygame.quit()
			sys.exit()
		if ourevent.type == KEYDOWN:
			if ourevent.key == K_RIGHT and hero.rect.x < RIGHT_BOUND:
				hero.rect.x += 5
			if ourevent.key == K_LEFT and hero.rect.x > LEFT_BOUND:
				hero.rect.x -= 5
		if ourevent.type == KEYUP:
			if ourevent.key == K_SPACE:
				fire()

	updateEnemies()
	updateMissiles()
	hero.render()

	if random.randint(0, 100) < 3:
		bomb()
	
	checkCollisions()

	if len(enemies) == 0:
		reset_game()

	pygame.display.update()
	pygame.time.delay(5)

