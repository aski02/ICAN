# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

from datasets import generate_data
from ican import causal_inference

# -------------------------------------------------------------------------
# Computing accuracy and custom score
# -------------------------------------------------------------------------

def compute_accuracy(dim_reduction, neighbor_percentage, iterations, kernel, variance_threshold, independence_threshold, regression_method, independence_method, min_distance, min_projection):
    true_structures = ["X->Y", "X<-T->Y", "X<-T->Y", "Y->X", "X<-T->Y", "Y->X", "X->Y", "X<-T->Y", "X->Y", "X<-T->Y"]
    structures = []
    correct = 0
    counter = 0
    score = 0

    for i in range(6, 6 + len(true_structures)): 
        T, X, Y = generate_data(40, i)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        _, var, _, _, _, structure, p1, p2 = causal_inference(X, Y, dim_reduction, neighbor_percentage, iterations, kernel, variance_threshold, independence_threshold, regression_method, independence_method, min_distance, min_projection)
        structures.append(structure)

        if structure == true_structures[counter]:
            correct += 1
            score += 1
        elif structure == "X<-T->Y":
            if true_structures[counter] == "X->Y" and p1 >= independence_threshold and var < 1.0:
                score += 0.5
            elif true_structures[counter] == "Y->X" and p2 >= independence_threshold and var > 1.0:
                score += 0.5
        elif structure == "X->Y":
            if true_structures[counter] == "X<-T->Y":
                score += 0.5
        elif structure == "Y->X":
            if true_structures[counter] == "X<-T->Y":
                score += 0.5
        counter += 1

    accuracy = correct / len(true_structures)
    accuracy_score = score / len(true_structures)
    return (accuracy, accuracy_score, structures)
