
from numpy import *


def sigmod(x):
    return 1.0/(1.0 + exp(-x))

def dsigmod(y):
    return y * (1.0 - y)


def tanh(x):
    return math.tanh(x)

def dtanh(y):
    return 1.0 - y*y


class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        self.input = input + 1      #add 1 for bias node
        self.hidden = hidden
        self.output = output
        # set up array of 1 dimension for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        # create randomized weights
        self.wih = random.randn(self.input, self.hidden)
        self.who = random.randn(self.hidden, self.output)

        # create arrays of 0 for changes
        self.cih = zeros((self.input, self.hidden))
        self.cho = zeros((self.hidden, self.output))



    def feedForward(self, inputs):
        ''' 前馈计算各层结果

        :param inputs:
        :return:
        '''
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for h in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wih[i][h]
            self.ah[h] = sigmod(sum)

        # output activations
        for o in range(self.output):
            sum = 0.0
            for h in range(self.hidden):
                sum += self.ah[h] * self.who[h][o]
            self.ao[o] = sigmod(sum)

        return self.ao[:]


    def backPropagate(self, targets, tmpLambda, correct):
        ''' 反馈算法，计算delta，以及更新权重

        :param targets:
        :param tmpLambda:
        :param correct:
        :return:
        '''
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output
        for o in range(self.output):
            error = -(targets[o] - self.ao[o])
            output_deltas[o] = dsigmod(self.ao[o]) * error

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden
        for h in range(self.hidden):
            error = 0.0
            for o in range(self.output):
                error = output_deltas[o] * self.who[h][o]
            hidden_deltas[h] = dsigmod(self.ah[h]) * error

        # update the weights connecting hidden to output
        for h in range(self.hidden):
            for o in range(self.output):
                change = output_deltas[o] * self.ah[h]
                self.who[h][o] -= tmpLambda * change + correct * self.cho[h][o]
                self.cho[h][o] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for h in range(self.hidden):
                change = hidden_deltas[h] * self.ai[i]
                self.wih[i][h] -= tmpLambda * change + correct * self.cih[i][h]
                self.cih[i][h] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * ((targets[k] - self.ao[k]) ** 2)

        return error

    def train(self, cases, labels, iterations = 3000, Lambda = 0.05, correct=0.1):
        for i in range(iterations):
            error = 0.0
            for p in range(len(cases)):
                label = labels[p]
                case = cases[p]
                self.feedForward(case)
                error += self.backPropagate(label, Lambda, correct)

            if i % 500 == 0:
                print('error %-.5f' % error)


    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions


if __name__ == '__main__':
    cases = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    labels = [[0], [1], [1], [0]]
    nn = MLP_NeuralNetwork(2,5,1)
    nn.train(cases, labels, 10000, 0.05, 0.1)
    for case in cases:
        print(nn.feedForward(case))


