tooluse_pool = {'travel': '''[1] find_flights: Finds flights based on source, destination and date. Arguments: from_location (str), to_location (str), date (str) in YYYY-MM-DD format.
Format example: 'Action: find_flights, A, B, 2023-12-25 End Action'
Returns a list of flights, each represented as a dictionary with keys "from_location", "to_location" (destination), "date", and "price".
Example: [{"from_location": "A", "to_location": "B", "date": "2023-12-25", "price": 450}]
    Signature: find_flights(destination: str, date: str) -> List[Dict]
[2] book_hotel: Books a hotel based on location and preferences. Arguments: location (str), *preferences (variable number of str arguments).
Format example: 'Action: book_hotel, B, wifi, pool End Action'
Returns a list of hotels, each represented as a dictionary with keys "location", "preferences", "price_per_night", and "rating".
Example: [{"location": "A", "preferences": ["wifi", "pool"], "price_per_night": 120, "rating": 4}]
    Signature: book_hotel(location: str, *preferences: str) -> List[Dict]
[3] budget_calculator: Calculates the total budget for a trip. Arguments: flight_price (float), hotel_price_per_night (float), num_nights (int).
Format example: 'Action: budget_calculator, 500, 100, 4 End Action'
Returns the total budget (float).
    Signature: budget_calculator(flight_price: float, hotel_price_per_night: float, num_nights: int) -> float
[4] max: Finds the maximum value among the given arguments. Accepts variable number of float arguments.
Format example: 'Action: max, 142, 174, 10 End Action'
    Signature: max(*args: float) -> float
[5] min: Finds the minimum value among the given arguments. Accepts variable number of float arguments.
Format example: 'Action: min, 101, 12, 47, 14 End Action'
    Signature: min(*args: float) -> float
[6] sum: Sums the given arguments. Accepts variable number of float arguments.
Format example: 'Action: sum, 21, 74, 70, 2, 69 End Action'
    Signature: sum(*args: float) -> float
''',
'message': '''[1] convert_hex_to_ascii: Converts a hexadecimal string to ASCII. Arguments: hex_string (str)
Format example: 'Action: convert_hex_to_ascii, 1da5d41741 End Action'
    Signature: convert_hex_to_ascii(hex_string: str) -> str
[2] reverse_string: Reverses a string. Arguments: string (str)
Format example: 'Action: reverse_string, dad4ax1a4 End Action'
    Signature: reverse_string(string: str) -> str
[3] caesar_decode: Decodes a string using the Caesar cipher. Arguments: message (str), shift (int)
Format example: 'Action: caesar_decode, sadadawfxa42, 3 End Action'
    Signature: caesar_decode(message: str, shift: int) -> str
[4] string_length: Finds the length of a string. Arguments: string (str)
Format example: 'Action: string_length, 3dad47gvnt41 End Action'
    Signature: string_length(string: str) -> int
[5] minimum_value: Finds the minimum value from given arguments. Arguments: *args (variable number of arguments)
Format example: 'Action: minimum_value, 28841, 1547, 26547 End Action'
    Signature: minimum_value(*args) -> int/float
[6] maximum_value: Finds the maximum value from given arguments. Arguments: *args (variable number of arguments)
Format example: 'Action: maximum_value, 3132, 470, 896701, 7452 End Action'
    Signature: maximum_value(*args) -> int/float
''',
'dna': '''[1] count_nucleotides: Counts the occurrences of each nucleotide in a DNA sequence. Arguments: dna_sequence (str)
Format example: 'Action: count_nucleotides, DADAINAGCAGCD End Action'
    Signature: count_nucleotides(dna_sequence: str) -> dict
[2] transcribe_dna_to_mrna: Transcribes DNA sequence to mRNA. Arguments: dna_sequence (str)
Format example: 'Action: maximum_value, DADAINAGCAGCD End Action'
    Signature: transcribe_dna_to_mrna(dna_sequence: str) -> str
[3] translate_mrna_to_amino_acid: Translates mRNA sequence to a chain of amino acids. Arguments: mrna_sequence (str)
Format example: 'Action: translate_mrna_to_amino_acid, DADAINAGCAGCD End Action'
    Signature: translate_mrna_to_amino_acid(mrna_sequence: str) -> str
[4] find_max_nucleotide: Return the nucleotide (str) with the maximum count (int). Arguments: nucleotide_counts in the form of (k1, v1, k2, v2, ..., kn, vn)
Format example: 'Action: find_max_nucleotide, A, 4, G, 1, C, 5  End Action'
    Signature: find_max_nucleotide(*args) -> (str, int)
[5] is_valid_dna_sequence: Checks if the DNA sequence is valid. Arguments: dna_sequence (str)
Format example: 'Action: is_valid_dna_sequence, DADAINAGCAGCD End Action'
    Signature: is_valid_dna_sequence(dna_sequence: str) -> bool
[6] reverse_transcribe_mrna_to_dna: Reverse transcribes mRNA sequence to DNA. Arguments: mrna_sequence (str)
Format example: 'Action: reverse_transcribe_mrna_to_dna, ADAINAGCAGCD End Action'
    Signature: reverse_transcribe_mrna_to_dna(mrna_sequence: str) -> str
''',
'trade': '''[1] convert_currency: Converts the commodity price to local currency. Arguments: base_price (float), conversion_rate (float)
Format example: 'Action: convert_currency, 100 * 10, 1.5 End Action'
    Signature: convert_currency(base_price: float, conversion_rate: float) -> float
[2] calculate_tariff: Calculates the trade tariff based on the converted price. Arguments: price (float), tariff_rate (float, in %)
Format example: 'Action: calculate_tariff, 200, 8 End Action'
    Signature: calculate_tariff(price: float, tariff_rate: float) -> float
[3] estimate_final_value: Estimates the final trade value including the tariff. Arguments: price (float), tariff (float)
Format example: 'Action: estimate_final_value, 210, 32.3  End Action'
    Signature: estimate_final_value(price: float, tariff: float) -> float
[4] calculator: Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.
Format example: 'Action: calculator, 2 + (3 * 4) End Action'
    Signature: calculator(expression: str) -> float
[5] find_minimum: Finds the minimum value among the given arguments. Accepts variable number of float arguments.
Format example: 'Action: find_minimum, 41.2, 78, 910, 100 End Action'
    Signature: find_minimum(*args: float) -> float
[6] find_maximum: Finds the maximum value among the given arguments. Accepts variable number of float arguments.
Format example: 'Action: find_maximum, 84, 25.3, 97 End Action'
    Signature: find_maximum(*args: float) -> float
''',
'web': '''[1] click_url: Clicks on a URL. A clickable URL looks like [Clickable '<url_argument>'] in the webpage.
Arguments: url (str).
Format example: 'Action: click_url, '/example/potion/rare_potion' End Action'
Returns the rendered content of the webpage after clicking the URL showing on the current rendered page.
    Signature: click_url(url: str) -> str
[2] go_to_previous_page: Goes back to the previous page. It has no arguments.
Format example: 'Action: go_to_previous_page End Action'
After going back to the previous page, return the rendered content of the webpage.
    Signature: go_to_previous_page() -> str
[3] scroll_down: Scrolls down the view. It has no arguments.
Format example: 'Action: scroll_down End Action'
Returns the rendered content of the webpage after scrolling down.
    Signature: budget_calculator(flight_price: float, hotel_price_per_night: float, num_nights: int) -> float
[4] scroll_up: Scrolls up the view. It has no arguments.
Format example: 'Action: scroll_up End Action'
Returns the rendered content of the webpage after scrolling up.
    Signature: scroll_up() -> str
[5] view: Return the current view in string format of the rendered webpage. It has no arguments.
Format example: 'Action: view End Action'
Returns the rendered content of the webpage.
You should call this when you want to see the rendered content of the current webpage.
    Signature: view() -> str
[6] calculator: Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.
Format example: 'Action: calculator, 2 + (3 * 4) End Action'
    Signature: calculator(expression: str) -> float
'''}


#  这个文本包含了一个工具使用池的描述，列出了不同功能的可用命令和它们的参数格式。这些功能可以执行各种操作，例如：

# 1. 旅行相关功能（travel）
    # 查找航班（find_flights）：根据出发地、目的地和日期查找航班。
    # 预订酒店（book_hotel）：根据位置和偏好预订酒店。
    # 预算计算器（budget_calculator）：计算旅行的总预算。
# 2. 消息处理功能（message）
    # 十六进制转ASCII（convert_hex_to_ascii）：将十六进制字符串转换为ASCII。
    # 字符串逆转（reverse_string）：逆转一个字符串。
    # 凯撒密码解码（caesar_decode）：使用凯撒密码解码字符串。
# 3. DNA相关功能（dna）
    # 计数核苷酸（count_nucleotides）：统计DNA序列中每种核苷酸的出现次数。
    # 转录DNA到mRNA（transcribe_dna_to_mrna）：将DNA序列转录为mRNA。
# 4. 贸易相关功能（trade）
    # 货币转换（convert_currency）：将商品价格转换为当地货币。
    # 计算关税（calculate_tariff）：根据转换后的价格计算贸易关税。
# 5. 网页操作功能（web）
    # 点击URL（click_url）：点击网页中的链接。
    # 查看当前页面（view）：返回当前渲染的网页内容。
    # 目的
    # 这个工具池的结构和功能定义非常适合开发者或用户构建自动化脚本、应用程序或工作流，从而实现对各种操作的快速调用。这种设计使得用户能够方便地调用所需的功能，同时提供了清晰的参数格式和示例。

# 依据需求，用户可以通过将特定的指令格式化为上述样式，方便地进行各种任务。\


#  详细相关功能
# 1. 旅行相关功能（travel）
# find_flights: 查找航班，基于出发地、目的地和日期。

# 参数:
# from_location（出发地，字符串）
# to_location（目的地，字符串）
# date（日期，字符串，格式为 YYYY-MM-DD）
# 格式示例: 'Action: find_flights, A, B, 2023-12-25 End Action'
# 返回: 航班列表，每个航班是一个字典，包括:
# from_location（出发地）
# to_location（目的地）
# date（日期）
# price（价格）
# 示例返回: [{"from_location": "A", "to_location": "B", "date": "2023-12-25", "price": 450}]
# book_hotel: 根据位置和偏好预订酒店。

# 参数:
# location（位置，字符串）
# *preferences（可变数量的字符串参数，表示偏好）
# 格式示例: 'Action: book_hotel, B, wifi, pool End Action'
# 返回: 酒店列表，每个酒店是一个字典，包括:
# location（位置）
# preferences（偏好，列表）
# price_per_night（每晚价格）
# rating（评分）
# 示例返回: [{"location": "A", "preferences": ["wifi", "pool"], "price_per_night": 120, "rating": 4}]
# budget_calculator: 计算旅行的总预算。

# 参数:
# flight_price（航班价格，浮点数）
# hotel_price_per_night（酒店每晚价格，浮点数）
# num_nights（住宿天数，整数）
# 格式示例: 'Action: budget_calculator, 500, 100, 4 End Action'
# 返回: 总预算（浮点数）。
# max: 找出给定参数中的最大值。

# 接受可变数量的浮点数参数。
# 格式示例: 'Action: max, 142, 174, 10 End Action'
# min: 找出给定参数中的最小值。

# 接受可变数量的浮点数参数。
# 格式示例: 'Action: min, 101, 12, 47, 14 End Action'
# sum: 计算给定参数的总和。

# 接受可变数量的浮点数参数。
# 格式示例: 'Action: sum, 21, 74, 70, 2, 69 End Action'
# 2. 消息处理功能（message）
# convert_hex_to_ascii: 将十六进制字符串转换为 ASCII。

# 参数: hex_string（十六进制字符串，字符串）。
# 格式示例: 'Action: convert_hex_to_ascii, 1da5d41741 End Action'
# reverse_string: 反转字符串。

# 参数: string（字符串）。
# 格式示例: 'Action: reverse_string, dad4ax1a4 End Action'
# caesar_decode: 使用凯撒密码解码字符串。

# 参数:
# message（信息，字符串）
# shift（偏移量，整数）
# 格式示例: 'Action: caesar_decode, sadadawfxa42, 3 End Action'
# string_length: 计算字符串的长度。

# 参数: string（字符串）。
# 格式示例: 'Action: string_length, 3dad47gvnt41 End Action'
# minimum_value: 从给定参数中找出最小值。

# 接受可变数量的参数。
# 格式示例: 'Action: minimum_value, 28841, 1547, 26547 End Action'
# maximum_value: 从给定参数中找出最大值。

# 接受可变数量的参数。
# 格式示例: 'Action: maximum_value, 3132, 470, 896701, 7452 End Action'
# 3. DNA相关功能（dna）
# count_nucleotides: 统计 DNA 序列中每种核苷酸的出现次数。

# 参数: dna_sequence（DNA 序列，字符串）。
# 格式示例: 'Action: count_nucleotides, DADAINAGCAGCD End Action'
# transcribe_dna_to_mrna: 将 DNA 序列转录为 mRNA。

# 参数: dna_sequence（DNA 序列，字符串）。
# 格式示例: 'Action: transcribe_dna_to_mrna, DADAINAGCAGCD End Action'
# translate_mrna_to_amino_acid: 将 mRNA 序列翻译为氨基酸链。

# 参数: mrna_sequence（mRNA 序列，字符串）。
# 格式示例: 'Action: translate_mrna_to_amino_acid, DADAINAGCAGCD End Action'
# find_max_nucleotide: 返回出现次数最多的核苷酸及其计数。

# 参数: nucleotide_counts（参数以 k1, v1, k2, v2,... 的格式提供）。
# 格式示例: 'Action: find_max_nucleotide, A, 4, G, 1, C, 5 End Action'
# is_valid_dna_sequence: 检查 DNA 序列是否有效。

# 参数: dna_sequence（DNA 序列，字符串）。
# 格式示例: 'Action: is_valid_dna_sequence, DADAINAGCAGCD End Action'
# reverse_transcribe_mrna_to_dna: 将 mRNA 序列反转录为 DNA。

# 参数: mrna_sequence（mRNA 序列，字符串）。
# 格式示例: 'Action: reverse_transcribe_mrna_to_dna, ADAINAGCAGCD End Action'
# 4. 贸易相关功能（trade）
# convert_currency: 将商品价格转化为当地货币。

# 参数:
# base_price（基础价格，浮点数）
# conversion_rate（转换汇率，浮点数）
# 格式示例: 'Action: convert_currency, 100 * 10, 1.5 End Action'
# calculate_tariff: 根据转换后的价格计算贸易关税。

# 参数:
# price（价格，浮点数）
# tariff_rate（关税率，浮点数，以%为单位）
# 格式示例: 'Action: calculate_tariff, 200, 8 End Action'
# estimate_final_value: 估算包括关税的最终贸易价值。

# 参数:
# price（价格，浮点数）
# tariff（关税，浮点数）
# 格式示例: 'Action: estimate_final_value, 210, 32.3 End Action'
# calculator: 计算给定表达式并返回结果。

# 接受计算表达式作为输入。
# 格式示例: 'Action: calculator, 2 + (3 * 4) End Action'
# find_minimum: 在给定参数中找出最小值。

# 接受可变数量的浮点数参数。
# 格式示例: 'Action: find_minimum, 41.2, 78, 910, 100 End Action'
# find_maximum: 在给定参数中找出最大值。

# 接受可变数量的浮点数参数。
# 格式示例: 'Action: find_maximum, 84, 25.3, 97 End Action'
# 5. 网页操作功能（web）
# click_url: 点击 URL。可点击的 URL 在网页上看起来如下: [可点击'<url_argument>']

# 参数: url（字符串）。
# 格式示例: 'Action: click_url, '/example/potion/rare_potion' End Action'
# 返回: 点击后渲染的网页内容。
# go_to_previous_page: 返回到前一页。没有参数。

# 格式示例: 'Action: go_to_previous_page End Action'
# 返回: 返回到前一页后的渲染网页内容。
# scroll_down: 向下滚动视图。没有参数。

# 格式示例: 'Action: scroll_down End Action'
# 返回: 向下滚动后的渲染网页内容。
# scroll_up: 向上滚动视图。没有参数。

# 格式示例: 'Action: scroll_up End Action'
# 返回: 向上滚动后的渲染网页内容。
# view: 返回当前渲染网页的视图，以字符串格式展示。没有参数。

# 格式示例: 'Action: view End Action'
# 返回: 当前网页的渲染内容。