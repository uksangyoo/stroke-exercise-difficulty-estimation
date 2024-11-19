from utils import get_train_test_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter


'''
plotting code
'''
def get_X():
    '''
    This function generates a bunch of random points that we can color based on the causal effect
    '''
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

    X = np.vstack([X.T, x3, x4])

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


def plot_tree(interp):
    color_bar_huh = True
    cols = ['x','y','z','distance','x_squared']
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
    plt.show()


def plot_tree_trans(interp):
    color_bar_huh = True
    cols = ['x','y','z','distance','x_squared']
    group_dict = get_groups(interp.tree_model_.tree_, cols)

    X = get_X()
    data = pd.DataFrame(X, columns=cols)
    data['causal_effect'] = 0

    # Fill in the causal effects in terms of seconds
    for node, criteria in group_dict.items():
        query = ' and '.join(criteria)
        print(query)
        data.loc[data.query(query).index, 'causal_effect'] = interp.tree_model_.tree_.value[node][0,0]
    
    fig = plt.figure(figsize=(10, 8))  # Increase the figure size
    ax = plt.axes(projection='3d')

    # Adjust the aspect ratio if needed
    ax.set_box_aspect([2, 1, 1.5])  # Adjust values to provide more space around the plot
    fig.subplots_adjust(left=0.3, right=0.7,bottom=0.3, top=0.7)  # Adjust layout margins

    # fig.tight_layout()

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
    ax.set_xlim(-.3,.3)
    ax.set_xticks([-.3,-.2,-.1,0,.1,.2,.3])
    ax.set_xticklabels(['30','20','10','0','10','20','30'], ha='right')

    ax.set_zlabel('Height (cm)', labelpad=15)

    ax.set_ylim(0,.3)
    ax.set_yticks([0,.1,.2,.3])
    ax.set_yticklabels(['','10','20','30'], ha='left')
    ax.set_zlim(0,.4)
    ax.set_zticks([0,.1,.2,.3,.4])
    ax.set_zticklabels(['','10','20','30', '40'])

    PCM=ax.get_children()[12]
    if color_bar_huh:
        cbar = plt.colorbar(PCM, ax=ax, location='left', shrink=.7, pad=.05)
        cbar.set_label('Additional Time to Reach (s)')
        cbar.solids.set(alpha=1)

    unique_effects = data['causal_effect'].unique()
    indices_dict = {effect: data.index[data['causal_effect'] == effect].tolist() for effect in unique_effects}
    centroids = {effect: data.loc[indices_dict[effect], ['x','y','z']].mean().values for effect in unique_effects}
    group_offset_scales_x = [2, 1.5, 5, 2, 1.7, 2, 100]
    group_offset_scales_y = [2, 1.5, 2, 2, 1, 2, 100]
    group_offset_scales_z = [2, 2, 2, 0.6, 1, 1, 100]
    for effect, indices in indices_dict.items():
        centroid = centroids[effect]
        offset_x = centroid[0] * group_offset_scales_x[list(unique_effects).index(effect)]
        offset_y = centroid[1] * group_offset_scales_y[list(unique_effects).index(effect)]
        offset_z = centroid[2] * group_offset_scales_z[list(unique_effects).index(effect)]
        new_centroid = [centroid[0] + offset_x, centroid[1] + offset_y, centroid[2] + offset_z]
        data.loc[indices, 'x'] += offset_x
        data.loc[indices, 'y'] += offset_y
        data.loc[indices, 'z'] += offset_z
        ax.plot3D([centroid[0], new_centroid[0]], [centroid[1], new_centroid[1]], [centroid[2], new_centroid[2]], color='red', clip_on=False)

    ax.scatter3D(xs=data['x'], ys=data['y'], zs=data['z'], c=data['causal_effect'], alpha=.15, cmap='viridis', clip_on=False)
    plt.xlabel('Distance from Home Position (cm)', labelpad=15)
    plt.show()





if __name__ == '__main__':
    # the PID of the partipant to plot
    PID = 27
    datasets = get_train_test_data()

    np.random.seed(0)  # Set the random seed for reproducibility

    for dataset in datasets:
        # skip over datasets that aren't the target participant
        if dataset['pid'] != PID:
            continue

        # First, fit the causal forest
        model = CausalForestDML(discrete_treatment=True, random_state=0)
        X = np.concatenate([dataset['stroke_X_train'], dataset['neuro_X_train']])
        y = np.concatenate([dataset['stroke_y_train'], dataset['neuro_y_train']])

        T = np.zeros(X.shape[0])
        T[:len(dataset['stroke_X_train'])] = 1

        model.fit(Y=y, T=T, X=X)

        # Next train the interpreter
        interp = SingleTreeCateInterpreter(max_depth=3, min_samples_leaf=0.02)
        interp.interpret(model, X)

        # Finally, plot the tree
        plot_tree_trans(interp)


