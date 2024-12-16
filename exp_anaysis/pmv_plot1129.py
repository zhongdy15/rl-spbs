from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

def rough_pmv_ppd(tdb, v, rh, activity, garments):
    # input variables
    # tdb: dry bulb air temperature, [$^{\circ}$C]
    # v: average air speed, [m/s]
    # rh: relative humidity, [%]
    # activity: participant's activity description
    # garments: list of clothing items worn by the participant

    # 通过activity来计算met代谢率，代表人体活动代谢水平
    met = met_typical_tasks[activity]  # activity met, [met]

    # 通过garments来计算icl: 总的衣服热阻
    icl = sum(
        [clo_individual_garments[item] for item in garments]
    )  # calculate total clothing insulation

    # 计算动态衣服热阻
    clo = clo_dynamic(clo=icl, met=met)  # calculate the dynamic clothing insulation


    # 计算相对空气速度
    vr = v_relative(v=v, met=met)  # calculate the relative air velocity


    results = pmv_ppd(tdb=tdb, tr=tdb, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
    # print(results)
    # if results['pmv'] is np.nan:
    #     print(f"activity: {activity}, tdb: {tdb}, vr: {vr}, rh: {rh}, met: {met}, clo: {clo}")
    #     print(results)
    return results['pmv'], results['ppd']

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    tdb_low = 10
    tdb_high = 40
    interval = 0.1
    v = 0.15  # average air speed, [m/s]
    rh = 40  # relative humidity, [%]
    activity_list = ["Seated, quiet", "Filing, standing","Walking 2mph (3.2kmh)"]
    garments = ["Sweatpants", "T-shirt", "Shoes or sandals","Long sleeve shirt (thin)"]

    # 计算PMV和PPD
    all_pmv_list = [[] for _ in range(len(activity_list))]
    all_ppd_list = [[] for _ in range(len(activity_list))]
    for i in range(len(activity_list)):
        activity = activity_list[i]
        pmv_list = all_pmv_list[i]
        ppd_list = all_ppd_list[i]
        for tdb in np.arange(tdb_low, tdb_high + interval, interval):
            pmv, ppd = rough_pmv_ppd(tdb=tdb, v=v, rh=rh, activity=activity, garments=garments)
            pmv_list.append(pmv)
            ppd_list.append(ppd)

    # ploting the PMV curve for each activity
    for i in range(len(activity_list)):
        plt.plot(np.arange(tdb_low, tdb_high + interval, interval), all_pmv_list[i], label=activity_list[i])
        plt.xlabel('Dry bulb air temperature [$^{\circ}$C]')
        plt.ylabel('PMV')
        plt.legend()
    # draw a line for y=-1
    plt.plot(np.arange(tdb_low, tdb_high + interval, interval), [-1]*len(np.arange(tdb_low, tdb_high + interval, interval)), '--', color='gray')
    plt.plot(np.arange(tdb_low, tdb_high + interval, interval), [1] * len(np.arange(tdb_low, tdb_high + interval, interval)), '--', color='gray')
    # plt.xticks(np.arange(tdb_low, tdb_high + interval, interval))
    # grid on
    plt.grid(True)
    plt.show()
    # ploting the PPD curve for each activity
    for i in range(len(activity_list)):
        plt.plot(np.arange(tdb_low, tdb_high + interval, interval), all_ppd_list[i], label=activity_list[i])
        plt.xlabel('Dry bulb air temperature [$^{\circ}$C]')
        plt.ylabel('PPD')
        plt.legend()
    # draw a line for y=-1
    plt.plot(np.arange(tdb_low, tdb_high + interval, interval), [10] * len(np.arange(tdb_low, tdb_high + interval, interval)), '--', color='gray')
    # plt.xticks(np.arange(tdb_low, tdb_high + interval, interval))
    # grid on
    plt.grid(True)
    plt.show()


    # met_typical_tasks = {'Basketball': 6.3,
    #                      'Calisthenics': 3.5,
    #                      'Cooking': 1.8,
    #                      'Dancing': 3.4,
    #                      'Driving a car': 1.5,
    #                      'Driving, heavy vehicle': 3.2,
    #                      'Filing, seated': 1.2,
    #                      'Filing, standing': 1.4,
    #                      'Flying aircraft, combat': 2.4,
    #                      'Flying aircraft, routine': 1.2,
    #                      'Handling 100lb (45 kg) bags': 4.0,
    #                      'Heavy machine work': 4.0,
    #                      'House cleaning': 2.7,
    #                      'Lifting/packing': 2.1,
    #                      'Light machine work': 2.2,
    #                      'Pick and shovel work': 4.4,
    #                      'Reading, seated': 1.0,
    #                      'Reclining': 0.8,
    #                      'Seated, heavy limb movement': 2.2,
    #                      'Seated, quiet': 1.0,
    #                      'Sleeping': 0.7,
    #                      'Standing, relaxed': 1.2,
    #                      'Table sawing': 1.8,
    #                      'Tennis': 3.8,
    #                      'Typing': 1.1,
    #                      'Walking 2mph (3.2kmh)': 2.0,
    #                      'Walking 3mph (4.8kmh)': 2.6,
    #                      'Walking 4mph (6.4kmh)': 3.8,
    #                      'Walking about': 1.7,
    #                      'Wrestling': 7.8,
    #                      'Writing': 1.0}