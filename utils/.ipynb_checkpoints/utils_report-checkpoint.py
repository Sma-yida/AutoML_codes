#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:19:46 2022

@author: jxu
"""
import xlwt
import xlrd

def df_writer(df,start_row,start_col,ws):
    """
    将DataFrame直接写入xls的函数（依赖xlwt.Worksheet.Worksheet)

    Parameters
    ----------
    df : 
        TYPE: DataFrame
        DESCRIPTION: 待写入数据
    start_row : 
        TYPE：int
        DESCRIPTION： 起始行数
    start_col : 
        TYPE：int
        DESCRIPTION： 起始列数
    ws : 
        TYPE：xlwt.Worksheet.Worksheet
        DESCRIPTION：待写入的worksheet

    Returns
    -------
    ws : 
        TYPE: xlwt.Worksheet.Worksheet
        DESCRIPTION：已写入数据的worksheet 
    """
    column_name = list(df.columns)
    [nrow,ncol] = df.shape
    for i in range(ncol):
        ws.write(start_row,start_col+i,column_name[i])
    for c in range(ncol):
        for r in range(nrow):
            try:
                ws.write(start_row+1+r,start_col+c,float(df.iloc[r,c]))
            except:
                ws.write(start_row+1+r,start_col+c,str(df.iloc[r,c]))
    return ws

def df_writer_to_1_sheet(df,ws,comment=""):
    """
    将DataFrame直接写入xls的函数（依赖xlwt.Worksheet.Worksheet)

    Parameters
    ----------
    df : 
        TYPE: DataFrame
        DESCRIPTION: 待写入数据
    start_row : 
        TYPE：int
        DESCRIPTION： 待写入数据的起始行数
    start_col : 
        TYPE：int
        DESCRIPTION： 待写入数据的起始列数
    ws : 
        TYPE：xlwt.Worksheet.Worksheet
        DESCRIPTION：待写入的worksheet

    Returns
    -------
    ws : 
        TYPE: xlwt.Worksheet.Worksheet
        DESCRIPTION：已写入数据的worksheet 
    """
    ws.write(0,0,comment)
    ws = df_writer(df=df,start_row=2,start_col=2,ws=ws)
    return ws

def copy_sheet_to_wb(pre_copy_file,wb,sheet_name_prefix):
    """
    将xls文件中所有sheet复制到existing open xlwt.Workbook() 对象中

    Parameters
    ----------
    pre_copy_file : 
        TYPE: .xls
        DESCRIPTION: 待复制文件
    wb : 
        TYPE：xlwt.Workbook()
        DESCRIPTION： 待写入xlwt.Workbook()对象
    sheet_name_prefix : 
        TYPE：str
        DESCRIPTION： 新增sheet名前缀

    Returns
    -------
    copy的sheet数,list of sheet names
    """
    wb_tmp = xlrd.open_workbook(pre_copy_file)
    sheet_num = len(wb_tmp.sheets())
    sheet_nm = []
    for s in range(sheet_num):
        sheet_name=sheet_name_prefix+str(s)
        sheet_nm.append(sheet_name)
        ws = wb.add_sheet(sheet_name)
        table = wb_tmp.sheet_by_index(0)
        nrow = table.nrows
        ncol = table.ncols
        for r in range(nrow):
            for c in range(ncol):
                ws.write(r,c,table.cell_value(r,c))
    return sheet_num, sheet_nm
