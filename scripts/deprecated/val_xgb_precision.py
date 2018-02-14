# !/usr/bin/python
def evaluate(pred_file,  test_labels):
    ftrain = open(test_labels, 'r')
    fpred = open(pred_file, 'r')

    pred_total = 0
    pred_right = 0
    fuck = 0
    for (lt, lp) in zip(ftrain, fpred):
        delta = int(int(float(lp) + 0.5)) and int(lt[0])# int(0.955 / 1) = 0

        # print(lp, "->", int(int(float(lp) + 0.5)), " vs ", int(lt[0]))
        # print(int(lt[0]))      
        pred_total = pred_total + 1
        fuck = fuck + int(int(float(lp) + 0.5))
        pred_right = pred_right + delta        
    print(fuck)
    return(float(pred_right) / float(pred_total))

pr = evaluate("pred.txt", "feature.txt.train")
print("%f precision."%pr)