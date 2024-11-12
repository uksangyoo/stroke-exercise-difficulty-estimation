import econml
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def load_control_data():
    return pd.read_csv('../data/aggregated_pilot.csv')

def load_participant_data(pid):
    dfs = []
    for day in [1,2,3]:
        if os.path.isfile(f'../data/{pid}/visit_{day}.csv'):
            dfs.append(pd.read_csv(f'../data/{pid}/visit_{day}.csv'))
        
    return pd.concat(dfs)

def train_causal_tree(c_data, p_data, side):
    
    c_data['treatment'] = 0
    p_data['treatment'] = 1
    data = pd.concat([c_data.query(f"side=='{side}'"), p_data.query(f"side=='{side}'")])

    data['distance'] = data['x']**2 + data['y']**2 + data['z']**2
    data['x^2'] = data['x']**2

    X = data[['x','y','z', 'distance', 'x^2']]
    T = data['treatment']
    y = data['time_to_press']
    est = CausalForestDML(discrete_treatment=True, max_depth=3, min_samples_leaf=0.02)
    est.fit(y, T, X=X, W=None)

    interp = SingleTreeCateInterpreter(max_depth=3, min_samples_leaf=0.02)
    interp.interpret(est, X)
    
    return est, interp

def train_nn(c_data, p_data, side):
    c_data['treatment'] = 0
    p_data['treatment'] = 1
    data = pd.concat([c_data.query(f"side=='{side}'"), p_data.query(f"side=='{side}'")])

    data['distance'] = data['x']**2 + data['y']**2 + data['z']**2
    data['x^2'] = data['x']**2

    X = data[['x','y','z', 'distance', 'x^2']]
    T = data['treatment']
    y = data['time_to_press']

    # c_nn = MLPRegressor(random_state=1, max_iter=500).fit(X[T==0], y[T==0])
    # p_nn = MLPRegressor(random_state=1, max_iter=500).fit(X[T==1], y[T==1])

    # c_nn = svm.SVR().fit(X[T==0], y[T==0])
    # p_nn = svm.SVR().fit(X[T==1], y[T==1])

    # c_nn = KNeighborsRegressor(5).fit(X[T==0], y[T==0])
    # p_nn = KNeighborsRegressor(5).fit(X[T==1], y[T==1])

    # c_nn = DecisionTreeRegressor(max_depth=3).fit(X[T==0], y[T==0])
    # p_nn = DecisionTreeRegressor(max_depth=3).fit(X[T==1], y[T==1])

    c_nn = RandomForestRegressor(max_depth=3).fit(X[T==0], y[T==0])
    p_nn = RandomForestRegressor(max_depth=3).fit(X[T==1], y[T==1])

    return c_nn, p_nn



def evaluate(gt_c_data, gt_p_data, model, side):
    #first filter to the correct side
    gt_c_data = gt_c_data.query(f"side=='{side}'")
    gt_p_data = gt_p_data.query(f"side=='{side}'")

    #create the additional features for prediction
    gt_p_data['distance'] = gt_p_data['x']**2 + gt_p_data['y']**2 + gt_p_data['z']**2
    gt_p_data['x^2'] = gt_p_data['x']**2

    #calculate the empirical increase in difficulty
    for i, row in gt_p_data.iterrows():
        thresh = .05
        control_time = gt_c_data.query(f"(x < {row['x']} + {thresh}) and (x > {row['x']} - {thresh}) and (y < {row['y']} + {thresh}) and (y > {row['y']} - {thresh}) and (z < {row['z']} +  {thresh}) and (z > {row['z']} - {thresh})")['time_to_press'].median()
        personal_time = row['time_to_press']

        gt_p_data.at[i,'gt_effect'] = personal_time - control_time

    gt_p_data = gt_p_data.dropna()
    X = gt_p_data[['x','y','z','distance','x^2']]

    pred = model.effect(X)
    gt = gt_p_data['gt_effect'].values

    return pred, gt

def evaluate_nn(gt_c_data, gt_p_data, c_model, p_model, side):
    #first filter to the correct side
    gt_c_data = gt_c_data.query(f"side=='{side}'")
    gt_p_data = gt_p_data.query(f"side=='{side}'")

    #create the additional features for prediction
    gt_p_data['distance'] = gt_p_data['x']**2 + gt_p_data['y']**2 + gt_p_data['z']**2
    gt_p_data['x^2'] = gt_p_data['x']**2

    #calculate the empirical increase in difficulty
    for i, row in gt_p_data.iterrows():
        thresh = .05
        control_time = gt_c_data.query(f"(x < {row['x']} + {thresh}) and (x > {row['x']} - {thresh}) and (y < {row['y']} + {thresh}) and (y > {row['y']} - {thresh}) and (z < {row['z']} +  {thresh}) and (z > {row['z']} - {thresh})")['time_to_press'].median()
        personal_time = row['time_to_press']

        gt_p_data.at[i,'gt_effect'] = personal_time - control_time

    gt_p_data = gt_p_data.dropna()
    X = gt_p_data[['x','y','z','distance','x^2']]

    pred = p_model.predict(X) - c_model.predict(X)
    gt = gt_p_data['gt_effect'].values

    return pred, gt




'''
plotting code
'''
def get_X():
    NUM_SAMPLES = 10_000
    #sampled points
    x_scale, y_scale, z_scale = 60, 30, 40 # in centimeters
    x_offset, y_offset, z_offset = -30, 0, 0 # in centimeters
    max_distance, min_distance = 30, 10 # in centimeters

    # first generate random data within the correct bounds 
    data = np.random.random((NUM_SAMPLES,3)) * [x_scale, y_scale, z_scale] + [x_offset, y_offset, z_offset]
    #select those data that satisfy the space we are looking at
    X = data[(data[:,0]**2 + data[:,1]**2 < max_distance**2) & \
            (data[:,0]**2 + data[:,1]**2 > min_distance**2)]
    X = X/100
    x3 = X[:,0]**2 + X[:,1]**2 + X[:,2]**2
    x4 = X[:,0]**2

    X = np.vstack([X.T,x3,x4])

    return X.T

def get_groups(tree, covariates):
    '''
    This function takes the tree structure we learned, and converts it to a dictionary where
    the key is the leaf id (an integer), and the value is a list of boolean expressions that
    need to be satistfied to be a member of the leaf.
    
    tree - the sklearn tree structure
    covariates - a list of strings in order that correspond to the column names of a df
    '''
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    criteria_dict = {}
    
    stack = [(0, 0, [])]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth, criteria = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1, criteria + [f'{covariates[feature[node_id]]} <= {threshold[node_id]}']))
            stack.append((children_right[node_id], depth + 1, criteria + [f'{covariates[feature[node_id]]} > {threshold[node_id]}']))
        else:
            is_leaves[node_id] = True
            criteria_dict[node_id] = criteria
            
    
    return criteria_dict #return the lists that are not empty from criteria_array


def plot_tree(interp, pid):
    color_bar_huh = True
    cols = ['x','y','z','distance','xsquared']
    group_dict = get_groups(interp.tree_model_.tree_, cols)

    X = get_X()
    data = pd.DataFrame(X, columns=cols)
    data['causal_effect'] = 0

    #fill in the causal effects in terms of seconds
    for node, criteria in group_dict.items():
        query = ' and '.join(criteria)
        print(query)
        # print(interp.tree_model_.tree_.value[node][0,0])
        data.loc[data.query(query).index, 'causal_effect'] = interp.tree_model_.tree_.value[node][0,0]
    

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([2, 1, 1.33])
    fig.tight_layout()

    outline_color = '#aaaaaaee'
    ax.plot3D([.10, .10], [0,0], [0,.4], color=outline_color)
    ax.plot3D([-.10, -.10], [0,0], [0,.4], color=outline_color)
    ax.plot3D([.30, .30], [0,0], [0,.4], color=outline_color)
    ax.plot3D([-.30, -.30], [0,0], [0,.4], color=outline_color)
    ax.plot3D([-.30, -.10], [0,0], [.4,.4], color=outline_color)
    ax.plot3D([-.30, -.10], [0,0], [.0,.0], color=outline_color)
    ax.plot3D([.10, .30], [0,0], [.4,.4], color=outline_color)
    ax.plot3D([.10, .30], [0,0], [.0,.0], color=outline_color)
    theta= np.linspace(0, np.pi, 100)
    ax.plot3D(.1*np.cos(theta), .1*np.sin(theta), [0]*100, color=outline_color)
    ax.plot3D(.1*np.cos(theta), .1*np.sin(theta), [.4]*100, color=outline_color)
    ax.plot3D(.3*np.cos(theta), .3*np.sin(theta), [0]*100, color='#cccccc33')
    ax.plot3D(.3*np.cos(theta), .3*np.sin(theta), [.4]*100, color=outline_color)

    ax.scatter3D(xs=data['x'], ys=data['y'], zs=data['z'], c=data['causal_effect'], alpha=.15, cmap='viridis')


    ax.view_init(42, -70)
    # ax.view_init(5, -89)
    ax.set_xlim(-.3,.3)
    ax.set_xticks([-.3,-.2,-.1,0,.1,.2,.3])
    ax.set_xticklabels(['30','20','10','0','10','20','30'], ha='right')
    plt.xlabel('Distance from Home Position (cm)', labelpad=15)
    ax.set_zlabel('Height (cm)', labelpad=15)

    ax.set_ylim(0,.3)
    ax.set_yticks([0,.1,.2,.3])
    ax.set_yticklabels(['','10','20','30'], ha='left')
    ax.set_zlim(0,.4)
    ax.set_zticks([0,.1,.2,.3,.4])
    ax.set_zticklabels(['','10','20','30', '40'])

    PCM=ax.get_children()[12] #get the mappable, the 1st and the 2nd are the x and y axes
    if color_bar_huh:
        # cbar = plt.colorbar(PCM, ax=ax, ticks=[0.001, 0.2,.4,.6,.8, .999], location='left', shrink=.7, pad=.05)
        cbar = plt.colorbar(PCM, ax=ax, location='left', shrink=.7, pad=.05)
        # cbar.ax.set_yticklabels(['0%', '20%', '40%','60%','80%','100%'])
        # cbar.ax.set_yticklabels(['0.5', '1', '1.5','2','2.5','3'])
        cbar.set_label('Additional Time to Reach (s)')

        cbar.solids.set(alpha=1)

    plt.tight_layout()
    plt.savefig(f"trees/PID{pid}.png")


'''
End plotting code
'''

if __name__ == '__main__':
    metadata = pd.read_csv('../data/metadata.csv')
    participant_ids = [36, 21,23,25,26,27,28,29,30,31,32,33,35,36,37]

    truths = []
    predictions = []
    for PID in participant_ids:
        print(PID)
        side = metadata.query(f'pid == {PID}')['affected_arm'].values[0]

        #load data
        control_data = load_control_data()
        c_train, c_test = train_test_split(control_data, test_size=0.5)

        participant_data = load_participant_data(PID)
        p_train, p_test = train_test_split(participant_data, test_size=0.2)

        #train model
        forest, interp = train_causal_tree(c_train, p_train, side)
        pred, gt = evaluate(c_test, p_test, forest, side)

        plot_tree(interp, PID)
        interp.render(f'trees/tree{PID}', feature_names=['x','y','z', 'distance', 'xsquared'])

        # c_nn, p_nn = train_nn(c_train, p_train, side)
        # pred, gt = evaluate_nn(c_test, p_test, c_nn, p_nn, side)
        

        for p,t in zip(pred, gt):
            predictions.append(p)
            truths.append(t)

    print(r2_score(truths, predictions))


    


    



