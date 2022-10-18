from timeit import default_timer as timer

import catalogs
import neutrino


def print_time(func, *args, **kwargs):
    print(f'Executing \'{func.__name__}\' with {args} and {kwargs} ...')

    start = timer()
    func(*args, **kwargs)
    end = timer()

    print(f'\tdone: t = {end - start:.3f} s')


def prepare_all():
    print_time(catalogs.prepare_2mrs)
    print_time(catalogs.prepare_2mrsg)
    print_time(catalogs.prepare_cf2)
    print_time(catalogs.prepare_bzcat)
    print_time(catalogs.prepare_milliquas)


def simulate_all():
    print_time(catalogs.simulate, '2mrs')
    print_time(catalogs.simulate, '2mrsg')
    print_time(catalogs.simulate, 'cf2')
    print_time(catalogs.simulate, 'bzcat')


def all_graphs(catalog):
    print_time(neutrino.allsky, catalog, show=False)
    print_time(neutrino.gauss_graph, catalog, show=False)
    print_time(neutrino.hist_by, catalog, 'MAG ABS', show=False)
    print_time(neutrino.hist_by, catalog, 'DIST', show=False)
    print_time(neutrino.hist_by, catalog, 'NEU', show=False)
    print_time(neutrino.hist_by, catalog, 'BAR', show=False)
    print_time(neutrino.hist_by, catalog, 'MAG', show=False)
    print_time(neutrino.hist_by, catalog, 'Z', show=False)
    print_time(neutrino.hist_sum, catalog, show=False)

    if catalog[-2:] == '_s':
        print_time(neutrino.malmquist, catalog[:-2], show=False)

    elif catalog[-3:] == '_ss':
        print_time(neutrino.malmquist, catalog[:-3], show=False)


def all_graphs_main_catalogs():
    all_graphs('2mrs')
    all_graphs('2mrsg')
    all_graphs('cf2')
    all_graphs('bzcat')
    all_graphs('milliquas')


def all_graphs_sim_catalogs():
    all_graphs('2mrs_s')
    all_graphs('2mrs_ss')
    all_graphs('2mrsg_s')
    all_graphs('2mrsg_ss')
    all_graphs('cf2_s')
    all_graphs('cf2_ss')
    all_graphs('bzcat_s')
    all_graphs('bzcat_ss')


def main():
    pass


# TODO промоделировать пуассоновский процесс (обычный генератор) в кубе (сфере) некоторого радиуса
#  у каждого объекта будет абсолютная величина (генерировать через гауссиану)
#  посчитать видимую звездную
#  сделать два каталога: с селекцией и без (обрубить по видимой звездной)
#  число точек в искусственных каталогах должно быть равно числу точек в реальных
#  картинки:
#    гистограмма: поток нейтрино против расстояния
#    как в смещении Малмквиста (абсолютная от расстояния)
#    все те картинки, что мы делали для реальных каталогов (должно быть видно смещение)
#  ----------------------------------------------------------------------------------
#  должно получится 2 * 3 каталога (сделать сначала по bzcat)

# TODO (со звездочкой) исследовать каталог квазаров на однородные выборки (по Type, по Zcite)


if __name__ == '__main__':
    main()
