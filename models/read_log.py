import pandas as pd
#import plotly.express as px
import matplotlib.pyplot as plt

def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 

if __name__ == "__main__":
    log = log2df('/home/ahasssan/ahmed/temporal_efficient_training/save/VGG7/${args.model}/${args.lamb}_learnable_V_temporal_adjustment_Vth_1_sigmoid_vth/s${model}_training.log')
    epoch = log['ep']
    training_accuracy = log['train_top1']
    learnable_threshold = log['avg_vth']
    Validation_accuracy = log['valid_top1']
    Best_accuracy = log['Best_Accuracy']

    table = {
        'epoch': epoch,
        'training_accuracy': training_accuracy,
        'learnable_threshold': learnable_threshold,
        'Validation_accuracy':Validation_accuracy,
        'Best_accuracy':Best_accuracy,
    }

    variable = pd.DataFrame(table, columns=['epoch','training_accuracy','learnable_threshold', 'Validation_accuracy', 'Best_accuracy'])
    variable.to_csv('learnable_th_temporal_adjustment_Cosine_Vth_1_sigmoid.csv', index=False)
    
    #df = pd.read_csv('learnable_th_temporal_adjustment_Cosine_Vth_1_sigmoid.csv')
    #fig = px.line(df, x = 'epoch', y = 'Validation_accuracy', title='VGG7 Training Progress')
    #fig.show()

    #headers = ['epoch', 'Validation_accuracy']
    headers = ['epoch', 'learnable_threshold']
    df = pd.read_csv('learnable_th_temporal_adjustment_Cosine_Vth_1_sigmoid.csv', usecols=headers)
    print(df)
    df.set_index('epoch').plot()
    plt.savefig('per plot with mean update.png')
    plt.show()