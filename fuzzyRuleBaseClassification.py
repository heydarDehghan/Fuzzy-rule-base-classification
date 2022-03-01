from servises import Rule, Interval
import numpy as np


class FuzzyRuleBaseClassification:
    def __init__(self, data):
        self.data = data
        self.rule_list = []

    def make_rules(self):
        """
        in this function we try to make rules from labels of data that we have in dataset.
        :return:
        """
        # This loop in each iteration generate one rule
        for index, label in enumerate(self.data.class_list):
            rule = Rule()
            data_rule_label = self.data.trainData[self.data.trainLabel == label]
            # This loop in each iteration generate interval of feature for current rule
            for column_index in range(self.data.trainData.shape[1]):
                interval = Interval()
                interval.featureNumber = column_index
                min_column_value = np.amin(self.data.trainData[:, column_index], axis=0)
                max_column_value = np.amax(self.data.trainData[:, column_index], axis=0)
                label_mean_list = []
                for class_label in self.data.class_list:
                    label_data = self.data.trainData[self.data.trainLabel == class_label][:, column_index]
                    label_mean_list.append((class_label, np.mean(label_data)))
                label_mean_list.sort(key=lambda x: x[1])
                item = (label, np.mean(data_rule_label[:, column_index]))
                index_in_mean_list = label_mean_list.index(item)

                # calculate lower bound interval

                if index_in_mean_list == 0:
                    interval.lower_bound = min_column_value
                else:
                    previous_item = label_mean_list[index_in_mean_list - 1]
                    interval.lower_bound = item[1] - (
                            (abs(item[1] - previous_item[1]) / 2) + ((abs(item[1] - previous_item[1]) / 2) * 1 / 2))

                # calculate upper bound interval

                if index_in_mean_list == len(label_mean_list) - 1:
                    interval.upper_bound = max_column_value
                else:
                    next_item = label_mean_list[index_in_mean_list + 1]
                    interval.upper_bound = item[1] + (
                            (abs(item[1] - next_item[1]) / 2) + ((abs(item[1] - next_item[1]) / 2) * 1 / 3))

                rule.intervals.append(interval)

            self.rule_list.append(rule)

    def predict(self, test_data):
        predict_label = np.zeros((test_data.shape[0], len(self.rule_list)))
        for data_index, row in enumerate(test_data):
            data_predict_for_rules = [None] * len(self.rule_list)
            for rule_index, rule in enumerate(self.rule_list) :
                predict_for_rule = [None]*len(rule.intervals)
                for index, (interval, feature) in enumerate(zip(rule.intervals, row)):
                    status = 0
                    if interval.lower_bound <= feature <= interval.upper_bound :
                        status = 1

                    predict_for_rule[index] = status

                data_predict_for_rules[rule_index] = sum(predict_for_rule)

            predict_label[data_index] = np.array(data_predict_for_rules)

        return np.argmax(predict_label, axis=1)




