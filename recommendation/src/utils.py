def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""

    # 1. Удаление товаров, со средней ценой < 1$
    data = data[data['sales_value'] >= 1.]

    # 2. Удаление товаров со соедней ценой > 30$
    data = data[data['sales_value'] <= 30.]

    # 3. Придумайте свой фильтр
    # Отфильтруем данные, которым больше года.
    data = data[data['week_no'] >= data['week_no'].max() - 52]

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    top_5000 = popularity.sort_values('quantity', ascending=False).head(take_n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999

    return data