from src.forked_inference import build_inf_model, make_results_dfs
from src.data import IdxDatagen, IdxConfig


if __name__ == "__main__":

    input_path = '../data/input_data'
    training_path = '../data/training_bundle'


    CFG = IdxConfig(training_path, input_path, model='b0')
    # CFG.csv_path = '../data/input_data/annotations_PAC_AUS_SMALL.csv'
    idg = IdxDatagen(CFG)
    gen = idg.make_generator()

    inf_model = build_inf_model(CFG.model_path)

    results = inf_model.predict(gen, verbose=1)


    df = make_results_dfs(results, gen.ids)

    df.to_csv(CFG.output_csv_path, index=False)
    print("Saved results to {}".format(CFG.output_csv_path))