"""
股票池定义：S&P 500 核心 + Russell 1000 扩展
"""

# S&P 500 核心 (~107 只)
SP500_CORE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK-B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN",
    "ABT", "DHR", "NEE", "LIN", "TXN", "PM", "CMCSA", "VZ", "RTX", "HON",
    "AMGN", "UNP", "LOW", "NKE", "UPS", "INTC", "COP", "BMY", "SBUX", "BA",
    "CAT", "DE", "MS", "GS", "BLK", "AXP", "MDLZ", "ADI", "ISRG", "GILD",
    "PLD", "REGN", "SYK", "CB", "BKNG", "VRTX", "AMT", "TMUS", "CI",
    "MO", "DUK", "SO", "CL", "ZTS", "BDX", "CME", "TGT", "PNC", "ICE",
    "USB", "TFC", "SLB", "APD", "EOG", "WM", "EMR", "FDX", "ORLY", "NSC",
    "GD", "PSA", "AEP", "SRE", "MCK", "ADSK", "D", "ADP", "CCI", "KLAC",
    "MSCI", "FTNT", "AFL", "AIG", "SPG", "F", "GM", "HUM", "DOW",
]

# Russell 1000 扩展：在 SP500 基础上加入 ~400 只中大盘股
# 覆盖 Russell 1000 中 SP500 以外的主要成分
RUSSELL1000_EXTRA = [
    # 科技
    "CRWD", "SNOW", "DDOG", "ZS", "NET", "PANW", "TEAM", "MDB", "BILL", "OKTA",
    "HUBS", "VEEV", "ANSS", "CDNS", "SNPS", "NXPI", "MCHP", "ON", "SWKS", "QRVO",
    "MPWR", "ENPH", "SEDG", "FSLR", "TER", "KEYS", "TRMB", "PTC", "MANH", "PAYC",
    "PCTY", "SMAR", "DOCU", "ZEN", "COUP", "CRSP", "TWLO", "TTD", "ROKU", "U",
    "PINS", "SNAP", "MTCH", "BMBL", "ETSY", "EBAY", "CPRT", "CSGP", "FICO", "FIS",
    "FISV", "GPN", "WEX", "SQ", "PYPL", "AFRM", "SOFI", "HOOD", "COIN",
    # 医疗
    "DXCM", "ALGN", "HOLX", "IDXX", "IQV", "CRL", "WST", "MTD", "BIO", "TECH",
    "ILMN", "EXAS", "INCY", "ALNY", "SRPT", "SGEN", "MRNA", "BNTX", "ZBH", "BAX",
    "EW", "PODD", "RGEN", "RVTY", "HSIC", "XRAY", "NUVA", "NEOG", "TFX", "PEN",
    "HCA", "THC", "UHS", "DVA", "EHC", "ACHC", "AMEH", "OSH", "LNTH", "MEDP",
    # 金融
    "SCHW", "IBKR", "RJF", "LPLA", "MKTX", "CBOE", "NDAQ", "TROW", "IVZ", "BEN",
    "FRC", "SIVB", "WAL", "FHN", "ZION", "CMA", "KEY", "CFG", "HBAN", "RF",
    "FITB", "MTB", "NTRS", "STT", "ALLY", "DFS", "COF", "SYF", "EWBC", "WBS",
    "FNB", "SNV", "ASB", "OZK", "GBCI", "UMBF", "PNFP", "BOH", "CADE", "HWC",
    "WTFC", "SBNY", "PACW", "FLY", "VOYA", "LNC", "PFG", "UNM", "GL", "RGA",
    "EQH", "ATH", "AEL", "FAF", "FNF", "ESNT", "RDN", "MGIC", "NMIH",
    # 工业
    "VRSK", "IEX", "NDSN", "RBC", "ITT", "GNRC", "TTC", "MIDD", "SWK", "LECO",
    "ROP", "CARR", "OTIS", "TT", "IR", "XYL", "WTS", "MWA", "FBIN", "AAON",
    "AOS", "LII", "SNA", "GGG", "SITE", "TREX", "AZEK", "POOL", "WSO",
    "FAST", "GRMN", "ROK", "AME", "ZBRA", "BR", "JKHY", "TYL", "CGNX", "NOVT",
    "COHR", "MKSI", "ENTG", "LRCX", "AMAT", "KLAC",
    "DAL", "UAL", "LUV", "ALK", "JBLU", "SAVE", "AAL",
    "ODFL", "SAIA", "XPO", "CHRW", "JBHT", "KNX", "LSTR", "WERN",
    # 消费
    "CMG", "DPZ", "WING", "SHAK", "DNUT", "SBH", "EL", "COTY",
    "DECK", "CROX", "SKX", "HBI", "PVH", "GOOS", "TPR", "RL",
    "DLTR", "DG", "FIVE", "OLLI", "ULTA", "RH", "WSM", "TSCO",
    "AZO", "AAP", "GPC", "LKQ", "MNST", "CELH", "SAM", "TAP", "STZ",
    "KDP", "COKE", "KHC", "HSY", "SJM", "MKC", "HRL", "CAG", "CPB",
    "TSN", "JJSF", "POST", "LW", "USFD", "SYY", "PFGC",
    "MAR", "HLT", "H", "WH", "IHG", "CHH", "LVS", "MGM", "WYNN", "CZR",
    "NCLH", "RCL", "CCL", "EXPE", "ABNB", "TRIP",
    # 能源 & 材料
    "PXD", "FANG", "DVN", "MPC", "VLO", "PSX", "HES", "OXY", "APA", "CTRA",
    "EQT", "AR", "RRC", "SWN", "OVV", "CHK",
    "FCX", "NEM", "GOLD", "AEM", "WPM", "RGLD", "FNV",
    "NUE", "STLD", "RS", "CMC", "ATI", "CLF", "X", "AA", "CENX",
    "CE", "EMN", "RPM", "PPG", "SHW", "AXTA", "ECL", "DD", "ALB", "LTHM",
    "IFF", "FMC", "CF", "MOS", "NTR",
    # 公用事业 & REITs
    "ED", "EIX", "ETR", "FE", "PPL", "CMS", "WEC", "ES", "AEE", "LNT",
    "EVRG", "PNW", "NRG", "VST", "OGE", "ATO", "NI", "SWX",
    "O", "NNN", "WPC", "STAG", "ADC", "STOR", "SRC", "BNL",
    "EQR", "AVB", "ESS", "UDR", "MAA", "CPT", "INVH", "AMH",
    "DLR", "EQIX", "ARE", "BXP", "SLG", "VNO", "HIW", "KRC",
    "VICI", "GLPI", "RYN", "PLD",
    # 通信 & 媒体
    "CHTR", "LBRDA", "WBD", "PARA", "FOX", "FOXA", "NWSA", "NWS",
    "DIS", "LYV", "MTCH", "EA", "TTWO", "RBLX", "SE",
]

RUSSELL1000 = list(set(SP500_CORE + RUSSELL1000_EXTRA))
