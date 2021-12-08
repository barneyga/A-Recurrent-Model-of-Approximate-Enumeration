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
    
    x_axis = list(df.groupby(['num_dots']).describe()['prediction']['std'].index)
    y_axis = list(df.groupby(['num_dots']).describe()['prediction']['std'].values)
    fig = px.line(x=x_axis, y=y_axis, title='Standard Deviation',
              labels={'x': 'Number of Dots', 'y': 'Standard Deviation of Prediction'})
    fig.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model_name", help="echo the string you use here")
  args = parser.parse_args()
  main(args.model_name)
