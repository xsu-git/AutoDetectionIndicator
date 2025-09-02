#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/4/11 14:13     Xsu         1.0         日志输出工具
'''

import logging
from colorama import init, Fore, Style
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
    }

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        record.message = record.getMessage()
        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
        record.asctime = f"{Fore.CYAN}{record.asctime}{Style.RESET_ALL}"
        record.funcName = f"{Fore.BLUE}{record.funcName}{Style.RESET_ALL}"
        record.message = f"{level_color}{record.message}{Style.RESET_ALL}"
        return super().format(record)

class BotsLogger:
    '''
    自定义日志输出类
    '''
    def __init__(self,specify_log_file:str = "app.log"):
        self.local_log_path = Path(__file__).parent.parent.absolute() / "logs" / specify_log_file
        self.bots_logger = self._set_up()

    def _set_up(self):
        init(autoreset=True)
        bots_logger = logging.getLogger("bots")
        bots_logger.setLevel(logging.INFO)
        base_format = "%(asctime)s - [%(levelname)s] [%(funcName)s] - %(message)s"
        base_date_format = '%Y-%m-%d %H:%M:%S'

        handler = TimedRotatingFileHandler(
            filename=self.local_log_path,
            when='midnight',
            interval=1,
            backupCount=30
        )
        formatter = logging.Formatter(base_format, datefmt=base_date_format)
        handler.setFormatter(formatter)


        console_handler = logging.StreamHandler()
        console_formatter = ColorFormatter(fmt=base_format,datefmt=base_date_format)
        console_handler.setFormatter(console_formatter)

        bots_logger.addHandler(handler)
        bots_logger.addHandler(console_handler)

        return bots_logger

    def init(self):
        return self.bots_logger
