import numpy as np

def get_group(train_df, target: str, target_dict):
    """
    target: user_id, track_id, album_id
    target_dict: self.user_dict, self.item_dict, self.album_dict
    Return
    group_dict: A dictionary where a key is an index of user/item and a value is groupid where a user/item belongs to.
    """
    # create user2group dict based on the fairness metric
    if target == 'user_id':
        bins = np.array([1,100,1000])
    elif target == 'track_id':
        bins = np.array([1,10,100,1000])
    else:
        bins = np.array([1,100,1000,10000])
    temp = train_df.groupby(target,as_index=True, sort=False)[['user_track_count']].sum()
    temp.index = temp.index.map(target_dict)
    temp['bin_index'] = np.digitize(temp.values.reshape(-1), bins) - 1
    group_dict = temp['bin_index'].to_dict()#.rename(index=target_dict)
    return group_dict