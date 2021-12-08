import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import argparse
import os


def main(model_name):
    # path = os.path.join('models',model_name,model_name+'_test.csv')
    path = os.path.join('models',model_name,model_name+'_SMALL.csv')
    # can probably use either matplotlib or seaborn as well.
    cols = ['image_id', 'num_dots', 'timestep', 'prediction', 'absolute_error']
    df = pd.read_csv(path, usecols = cols)
    
    #index_list = list(df.groupby(['num_dots', 'timestep']).describe()['final_prediction']['mean'].index)
    #x_axis, color = zip(*index_list)
    #y_axis = list(df.groupby(['num_dots', 'timestep']).describe()['final_prediction']['mean'].values)
    #fig = px.line(x=x_axis, y=y_axis, color=color, title='Final Predictions',
    #              labels={'x': 'Number of Dots', 'y': 'Average Final Prediction'})
    #fig.show()

    # prediction_save_path = os.path.join('models',model_name,model_name+'_predictionplot.png')
    # fig.write_image(prediction_save_path)
    
    index_list = list(df.groupby(['num_dots', 'timestep']).describe()['prediction']['mean'].index)
    x_axis, color = zip(*index_list)
    y_axis = list(df.groupby(['num_dots', 'timestep']).describe()['prediction']['mean'].values)
    fig = px.line(x=x_axis, y=y_axis, color=color, title='Predictions',
                  labels={'x': 'Number of Dots', 'y': 'Average Prediction'})
    fig.show()


    index_list = list(df.groupby(['num_dots', 'timestep']).describe()['absolute_error']['mean'].index)
    x_axis, color = zip(*index_list)
    y_axis = list(df.groupby(['num_dots', 'timestep']).describe()['absolute_error']['mean'].values)
    fig = px.line(x=x_axis, y=y_axis, color=color, title='Absolute Errors',
                  labels={'x': 'Number of Dots', 'y': 'Average Absolute Error'})
    fig.show()
    # error_save_path = os.path.join('models',model_name,model_name+'_errorplot.png')
    # fig.write_image(error_save_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model_name", help="echo the string you use here")
  args = parser.parse_args()
  main(args.model_name)
