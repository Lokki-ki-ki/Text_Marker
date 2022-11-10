import pickle
from dataeng import DataEng


def greeting():
    username = input("Welcome to Linguistic! Please enter your username: \n")
    print("Hi " + username + "! It's my pleasure to serve you:)")
    print("\n")


def result(predictions):
    result_list = []
    for pred in predictions:
        result = pred // 0.5 * 0.5
        if (pred - result) > 0.25:
            result += 0.5
        if result < 1.0:
            result = 1.0
        if result > 5.0:
            result = 5.0
        result_list.append(result)
    return result_list


def main():
    model = pickle.load(open('Models_sav\Decision_Tree_vocab.sav', "rb"))
    reply = True
    greeting()
    while reply:
        text = input(
            "Please enter the text, or type Leave to exit the program.\n")
        if text != 'Leave' and text != 'leave':
            print("Grading your texts, please wait...\n")
            data = DataEng(text).Engineering()
            prediction = model.predict(data)
            print(result(prediction))
        else:
            reply = False

    print("Thank you for using the service of Linguistic. See you next time:)")
    return


if __name__ == '__main__':
    main()
