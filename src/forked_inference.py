from tensorflow.keras.models import Model, load_model
import numpy as np
import pandas as pd

# from data import IdxDatagen, IdxConfig

def build_inf_model(weights_path):
    model = load_model(weights_path, compile=False)


    inf_model = Model(inputs=model.inputs,
                      outputs=[
                               model.get_layer('full_dense').output,
                               model.get_layer('reduced_dense').output,
                               model.get_layer('avg_pool').output,
                               model.get_layer('reduced_avg_pool').output
                      ])

    return inf_model


def make_results_dfs(results, pths):
    n_classes = np.shape(results[0])[1]
    full_vector_length = np.shape(results[2])[1]
    reduced_vector_length = np.shape(results[3])[1]


    # columns for softmax vectors for both full (results[0]) and reduced (results[1]) forks
    fs_cols = []
    rs_cols = []
    for i in range(n_classes):
        fs_cols.append('full_softmax_{}'.format(i))
        rs_cols.append('reduced_softmax_{}'.format(i))


    # columns for feature vector from full fork (results[2])
    f_vector_cols = []
    for i in range(full_vector_length):
        f_vector_cols.append('full_feature_vector_{}'.format(i))

    # columns for feature vector from reduced fork (results[3])
    r_vector_cols = []
    for i in range(reduced_vector_length):
        r_vector_cols.append('reduced_feature_vector_{}'.format(i))


    df_cols = ['quadratid'] + fs_cols + rs_cols + f_vector_cols + r_vector_cols
    df = pd.DataFrame(columns=df_cols)

    # populate df columns
    df['quadratid'] = pths
    df[fs_cols] = results[0]
    df[rs_cols] = results[1]
    df[f_vector_cols] = results[2]
    df[r_vector_cols] = results[3]

    return df


# if __name__ == "__main__":
#
#     input_path = '../data/input_data'
#     training_path = '../data/training_bundle'
#
#
#     CFG = IdxConfig(training_path, input_path, model='b0')
#     # CFG.csv_path = '../data/input_data/annotations_PAC_AUS_SMALL.csv'
#     idg = IdxDatagen(CFG)
#     gen = idg.make_generator()
#
#     inf_model = build_inf_model(CFG.model_path)
#
#     results = inf_model.predict(gen, verbose=1)
#
#
#     df = make_results_dfs(results, gen.ids)
#
#     df.to_csv(CFG.output_csv_path, index=False)
#     print("Saved results to {}".format(CFG.output_csv_path))