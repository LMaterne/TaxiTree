from TaxiTree import World, RandomTree


CYCLES = 10
DRAW_EVERY_N_CYCLE = 1


def main():
    world = World(
        [RandomTree(i, 0, i, energy=50) for i in range(5, 30, 5)],
        (30, 10)
    )
    world.draw()
    for cycle in range(CYCLES):
        world.update()
        if cycle % DRAW_EVERY_N_CYCLE == 0:
            world.draw()


if __name__ == "__main__":
    main()
