import pandas as pd
import random


class Dataset():
    def __init__(self):
        self.df = self.init_dataset()

    def init_dataset(self):
        df = pd.read_csv('RefractiveIndicesRaw - Sheet1.csv',
                         names=['refractive_ind', 'material', 'info'])
        new_df = pd.DataFrame()
        for index, row in df.loc[df.refractive_ind.str.contains('-')].iterrows():
            new_df = pd.concat([new_df, Dataset.__mod_frame(row)])
        df = pd.concat([df, new_df]).loc[~df.refractive_ind.astype(
            str).str.contains('-')]
        df = df.loc[~df.refractive_ind.astype(str).str.contains('–')]

        new_df = pd.DataFrame()
        for i, count in enumerate(df['refractive_ind'].value_counts()):
            if count > 1:
                select = random.randint(0, count-1)
                vals = df.loc[df['refractive_ind'] ==
                              df['refractive_ind'].value_counts().index[i]]
                save = vals.copy().reset_index(drop=True).iloc[[select]]
                new_df = pd.concat([new_df, save])
            else:
                val = df.loc[df['refractive_ind'] ==
                             df['refractive_ind'].value_counts().index[i]]
                new_df = pd.concat([new_df, val])

        new_df.reset_index(drop=True)
        new_df['refractive_ind'] = new_df['refractive_ind'].astype(float)
        return new_df

    def get_material(self, n: float) -> tuple:
        return tuple(self.df.iloc[(self.df['refractive_ind']-n).abs().argsort()[:1]].values[0])

    @staticmethod
    def __mod_frame(x):

        refractive_ind = x['refractive_ind'].split('-')
        # print(refractive_ind)
        if len(refractive_ind) > 1:
            lower = int(float(refractive_ind[0]) *
                        (10**len(refractive_ind[0].split('.')[1])))
            upper = int(float(refractive_ind[1]) *
                        (10**len(refractive_ind[0].split('.')[1])))
            elem_range = list(range(lower, upper+1))
            rows = []
            for ref_in in [elem/(10**len(refractive_ind[0].split('.')[1])) for elem in list(range(lower, upper+1))]:
                rows.append(pd.Series([ref_in, x['material'], x['info']], index=[
                            'refractive_ind', 'material', 'info']))
            return pd.DataFrame(rows)
        else:
            return pd.Series([x['refractive_ind'], x['material'], x['info']], index=['refractive_ind', 'material', 'info'])
