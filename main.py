from servises import *
from fuzzyRuleBaseClassification import FuzzyRuleBaseClassification as Fz


def add_interval(ax, xdata, ydata, caps="  "):
    line = ax.add_line(mpl.lines.Line2D(xdata, ydata))
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 24,
        'color': line.get_color()
    }
    a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), **anno_args)
    a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), **anno_args)
    return (line,(a0,a1))

if __name__ == "__main__":
    ''' '''
    raw_data = load_data('DataSets/hcvdat0.csv', array=True)
    # raw_data[:, 1] = [x[0] for x in raw_data[:, 1]]
    data = Data(data=raw_data, delete_column=[3], data_range=(2, -1), label_range=1, bias=False, normal=False)

    model = Fz(data)
    model.make_rules()




    predictt = model.predict(data.trainData)
    acct = accuracy_metric(data.trainLabel, predictt)
    print(f'accuracy train : {acct}')

    predict = model.predict(data.testData)
    acc = accuracy_metric(data.testLabel, predict)
    print(f'accuracy test : {acc}')


    # for x in model.rule_list:
    #     inte = x.intervals[0]
    #     print(inte.lower_bound)
    #     print(inte.upper_bound)
    #     print('---------------')
