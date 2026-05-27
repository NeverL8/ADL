import awkward

def normalize_x(dataset):
    x=dataset["data"][:,1:2,:]
    x_std=awkward.std(x)
    x_mean=awkward.mean(x)
    norm_x=(x-x_mean)/x_std
    return norm_x

def normalize_y(dataset):
    y=dataset["data"][:,2:3,:]
    y_std=awkward.std(y)
    y_mean=awkward.mean(y)
    norm_y=(y-y_mean)/y_std
    return norm_y

def normalize_time(dataset):
    times = dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                               # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    # make times relative per event
    t_rel = times- awkward.min(times,axis=2)
    #print(t_rel[0])
    # global normalization
    mean_t = awkward.mean(awkward.flatten(t_rel))
    std_t = awkward.std(awkward.flatten(t_rel))

    norm_times = (t_rel - mean_t) / std_t
    return norm_times


def denormalize(pred, target_x_mean, target_x_std, target_y_mean, target_y_std):

    pred[:, 0] = (
        (pred[:, 0] * target_x_std)
        + target_x_mean
    )

    pred[:, 1] = (
        (pred[:, 1] * target_y_std)
        + target_y_mean
    )

    return pred
