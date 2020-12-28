import matplotlib.pyplot as plt


def plot_legend(datas, legends=None, line_form=None, file_name=None, fill=None,
                title=None, xlabel=None, ylabel=None):
    """
    參數:
        datas(list of list): list of y_data 
        
    可選參數:
        legends(list): 圖例名稱, 可穿插None指定哪些線不畫圖例
        line_form: 設定線條樣式
        file_name(str): 若file_name不為None，將圖表存檔為檔名
        fill(tuple of 4): 可以指定兩條線之間塗色，例(x, y_1, y_2, 'pink')
        title, xlabel, ylabel are string
    """
    plt.figure()
    for idx, data in enumerate(datas):
        args, kwargs  = list(), dict()
        if line_form:
            args.append(line_form[idx])
        if legends and legends[idx]!=None:
            kwargs['label'] = legends[idx]
        plt.plot(data[0], data[1],*args, **kwargs)
    if fill:
        plt.fill_between(*fill[:3], color=fill[3])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if legends:
        plt.legend()
    if file_name:
        plt.savefig(file_name, bbox_inches='tight') #存檔，第二個參數表示把圖表外多餘的空間刪除
    plt.show()

if __name__ == '__main__':
    # caller exmple
    x = [1,2,3,4,5,6,7,8]
    y_1 = [1,4,9,16,25,36,49,64]
    y_2 = [1,8,27,64,125,216,343,512]
    datas = [(x,y_1),(x,y_2)]
    plot_legend(datas, fill=(x,y_1,y_2,'pink'))
    plot_legend(datas, legends = ['x_square', 'x_cube'])
    plot_legend(datas, legends = ['x_square', 'x_cube'], line_form = ['r-.^', 'g--*'])
    plot_legend(datas, legends = [None, 'x_cube'], line_form = ['r-.^', 'g--*'])