"""
Helper classes and methods used for streaming of tabular data.
"""

from __future__ import annotations

import errno
import heapq
import logging
import warnings
from typing import Callable, Generator, Iterator

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot.tabular_data import (
    BufferType,
    TabularDataReader,
    TabularDataWriter,
)

LOGGER = logging.getLogger(__name__)


@typechecked
class JoinedTabularDataReader(TabularDataReader):
    """
    Handles data from multiple tabular data sources, joining them horizontally.

    Attributes:
    -----------
        readers : list[TabularDataReader]
            A list of 'TabularDataReader' objects representing the individual
            data sources.
    """

    readers: list[TabularDataReader]

    def __init__(self, readers: list[TabularDataReader]):
        self.readers = readers

    def get_column_names(self) -> list[str]:
        return sum([reader.get_column_names() for reader in self.readers], [])

    def get_column_types(self) -> list:
        return sum([reader.get_column_types() for reader in self.readers], [])

    def _subset_columns(self, column_names: list[str] | None) -> list[list[str] | None]:
        if column_names is None:
            return [None for _ in self.readers]
        return [
            [
                column_name
                for column_name in reader.get_column_names()
                if column_name in column_names
            ]
            for reader in self.readers
        ]

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        subset_column_lists = self._subset_columns(columns)
        df = pd.concat(
            [
                reader.read(columns=subset_columns)
                for reader, subset_columns in zip(self.readers, subset_column_lists)
            ],
            axis=1,
        )
        return df if columns is None else df[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        subset_column_lists = self._subset_columns(columns)
        iterators = [
            reader.get_chunked_data_iterator(
                chunk_size=chunk_size, columns=subset_columns
            )
            for reader, subset_columns in zip(self.readers, subset_column_lists)
        ]

        while True:
            try:
                chunks = [next(iterator) for iterator in iterators]
            except StopIteration:
                break
            df = pd.concat(chunks, axis=1)
            yield df if columns is None else df[columns]


@typechecked
class ComputedTabularDataReader(TabularDataReader):
    """
    A subclass of TabularDataReader that allows the computation of a specific
    column that is joined horizontally to the columns of the reader.

    Attributes:
    -----------
        reader : TabularDataReader
            The underlying reader object.
        column : str
            The name of the column to compute.
        dtype : np.dtype | pa.DataType
            The data type of the computed column.
        func : Callable
            A function to apply to the existing columns of each chunk.
    """

    def __init__(
        self,
        reader: TabularDataReader,
        column: str,
        dtype: np.dtype,
        func: Callable,
    ):
        self.reader = reader
        self.dtype = dtype
        self.func = func
        self.column = column

    def get_column_names(self) -> list[str]:
        return self.reader.get_column_names() + [self.column]

    def get_column_types(self) -> list:
        return self.reader.get_column_types() + [self.dtype]

    def _reader_columns(self, columns: list[str] | None):
        # todo: performance: Currently, we need to read all columns, since we
        #  don't know what's needed in the computation. This could be made more
        #  efficient by letting the class know which columns those are.
        return None

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.reader.read(self._reader_columns(columns))
        # We need to compute the result column only in two cases:
        #   a) all columns are requested (columns = None)
        #   b) the computed column is requested explicitly
        if columns is None or self.column in columns:
            df[self.column] = self.func(df)
        return df if columns is None else df[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        iterator = self.reader.get_chunked_data_iterator(
            chunk_size=chunk_size, columns=self._reader_columns(columns)
        )

        while True:
            try:
                df = next(iterator)
            except StopIteration:
                break
            # See comments in `read` for explanation
            if columns is None or self.column in columns:
                df[self.column] = self.func(df)
            yield df if columns is None else df[columns]


@typechecked
class MergedTabularDataReader(TabularDataReader):
    """
    Merges data from multiple tabular data sources vertically into a single
    data source, ordering the rows (one by one) by the value of a priority
    column. I.e. for each output row, the row of the input readers with the
    highest value of the priority column is picked.

    Attributes:
    -----------
        readers : list[TabularDataReader]
            List of data readers to merge.
        priority_column : str
            Name of the priority column used for merging (highest value
            determines which reader to pick next).
        descending : bool
            Flag indicating whether the merge should be in descending order
            (default: True).
        reader_chunk_size : int
            Chunk size used when iterating over data readers (default: 1000).
        column_names : list[str]
            List of column names for the merged data.
        column_types : list
            List of column types for the merged data.
    """

    def __init__(
        self,
        readers: list[TabularDataReader],
        priority_column: str,
        descending: bool = True,
        reader_chunk_size: int = 1000,
    ):
        self.readers = readers
        self.priority_column = priority_column
        self.descending = descending
        self.reader_chunk_size = reader_chunk_size

        if len(readers) == 0:
            raise ValueError("At least one data reader is required")

        self.column_names = readers[0].get_column_names()
        self.column_types = readers[0].get_column_types()

        for reader in readers:
            if not reader.get_column_names() == self.column_names:
                raise ValueError("Column names do not match")

            if not reader.get_column_types() == self.column_types:
                raise ValueError("Column types do not match")

            if priority_column not in self.column_names:
                raise ValueError("Priority column not found")

    def get_column_names(self) -> list[str]:
        return self.column_names

    def get_column_types(self) -> list:
        return self.column_types

    def get_row_iterator(
        self,
        columns: list[str] | None = None,
        row_type: BufferType = BufferType.DataFrame,
        check_sorting: bool = True,
    ) -> Iterator[pd.DataFrame | dict | np.record]:
        # Define methods to iterate over dataframe, dicts or records and also
        # to get specific column values from each of those data structures
        def iterate_over_df(df: pd.DataFrame) -> Iterator:
            for i in range(len(df)):
                row = df.iloc[[i]]
                row.index = [0]
                yield row

        def get_value_df(row, col):
            return row[col].iloc[0]

        def iterate_over_dicts(df: pd.DataFrame) -> Iterator:
            dict = df.to_dict(orient="records")
            return iter(dict)

        def get_value_dict(row, col):
            return row[col]

        def iterate_over_records(df: pd.DataFrame) -> Iterator:
            records = df.to_records(index=False)
            return iter(records)

        # Set iteration function and get_value function depending on current
        # buffer type
        if row_type == BufferType.DataFrame:
            iterate_over_chunk = iterate_over_df
            get_value = get_value_df
        elif row_type == BufferType.Dicts:
            iterate_over_chunk = iterate_over_dicts
            get_value = get_value_dict
        elif row_type == BufferType.Records:
            iterate_over_chunk = iterate_over_records
            get_value = get_value_dict
        else:
            raise ValueError(
                f"ret_type must be 'dataframe', 'records' or 'dicts', not {row_type}"
            )

        # Get a row iterator from a chunked iterator
        def row_iterator_from_chunked(chunked_iter: Iterator) -> Iterator:
            for chunk in chunked_iter:
                for row in iterate_over_chunk(chunk):
                    yield row

        # Collect all iterators
        row_iterators = [
            row_iterator_from_chunked(
                reader.get_chunked_data_iterator(
                    chunk_size=self.reader_chunk_size, columns=columns
                )
            )
            for reader in self.readers
        ]

        # Use builtin merge function for merging row_iterators using a
        # priority queue internally (O(1) lookup, O(log N) insert)
        merged_iter = heapq.merge(
            *row_iterators,
            key=lambda row: get_value(row, self.priority_column),
            reverse=self.descending,
        )

        if not check_sorting:
            return merged_iter

        def checked_iter():
            # Note: directly yielding from this function is terribly slow, while
            # first wrapping the iteration and checking into a generator function
            # is much faster. I don't know why exactly - maybe it needs less saving
            # and restoring of context when returning to the generator from the
            # caller - but that's just wild guessing.
            last_value = None
            try:
                for row in merged_iter:
                    new_value = get_value(row, self.priority_column)
                    if last_value is None:
                        last_value = new_value
                    if self.descending and new_value > last_value:
                        raise ValueError(
                            f"New value {new_value} for {self.priority_column} exceeds "
                            f"previous value {last_value} but should be descending."
                        )
                    if not self.descending and new_value < last_value:
                        raise ValueError(
                            f"New value {new_value} for {self.priority_column} lower "
                            f"than previous value {last_value} but should be ascending."
                        )
                    last_value = new_value

                    yield row
            except OSError as e:
                # If the error was caused by too many open files (EMFILE) we
                # supply extra information to the user and re-raise the exeception
                # to be (possibly) handled by a further exception handler
                if e.errno == errno.EMFILE:
                    min = len(row_iterators)
                    LOGGER.info(
                        "Cannot open all necessary files simultaneously. Please \n\t"
                        "raise the limit imposed by the OS by using the `ulimit`\n\t"
                        f"command. Since at least {min} files are to be opened\n\t"
                        f"something like `ulimit -S -n {min + 10}` will probably\n\t"
                        "suffice."
                    )
                raise

        return checked_iter()

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        row_iterator = self.get_row_iterator(columns=columns)
        finished = False
        rows = []
        while not finished:
            try:
                row = next(row_iterator)
                rows.append(row)
            except StopIteration:
                finished = True
            if (finished and len(rows) > 0) or len(rows) == chunk_size:
                df = pd.concat(rows)
                df.reset_index(drop=True, inplace=True)
                yield df
                rows = []

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        row_iterator = self.get_row_iterator(columns=columns)
        rows = [row for row in row_iterator]
        df = pd.concat(rows)
        df.reset_index(drop=True, inplace=True)
        return df


@typechecked
def join_readers(readers: list[TabularDataReader]):
    return JoinedTabularDataReader(readers)


@typechecked
def merge_readers(
    readers: list[TabularDataReader],
    priority_column: str,
    descending: bool = True,
    reader_chunk_size: int = 1000,
):
    reader = MergedTabularDataReader(
        readers,
        priority_column,
        descending,
        reader_chunk_size=reader_chunk_size,
    )
    return reader.get_row_iterator()


@typechecked
class BufferedWriter(TabularDataWriter):
    """
    This class represents a buffered writer for tabular data. It allows
    writing data to a tabular data writer in batches, reducing the
    number of write operations.

    Attributes:
    -----------
    writer : TabularDataWriter
        The tabular data writer to which the data will be written.
    buffer_size : int
        The number of records to buffer before writing to the writer.
    buffer_type : TableType
        The type of buffer being used. Can be one of TableType.DataFrame,
        TableType.Dicts, or TableType.Records.
    buffer : pd.DataFrame or list of dictionaries or np.recarray or None
        The buffer containing the tabular data to be written.
        The buffer type depends on the buffer_type attribute.
    """

    writer: TabularDataWriter
    buffer_size: int
    buffer_type: BufferType
    buffer: pd.DataFrame | list[dict] | np.recarray | None

    def __init__(
        self,
        writer: TabularDataWriter,
        buffer_size=1000,
        buffer_type=BufferType.DataFrame,
    ):
        super().__init__(writer.columns, writer.column_types)
        self.writer = writer
        self.buffer_size = buffer_size
        self.buffer_type = buffer_type
        self.buffer = None
        # For BufferedWriters it is extremely important that they are
        # correctly initialized and finalized, so we make sure
        self.finalized = False
        self.initialized = False

    def __del__(self):
        if self.initialized and not self.finalized:
            warnings.warn(f"BufferedWriter not finalized (buffering: {self.writer})")

    def _buffer_slice(
        self,
        start: int = 0,
        end: int | None = None,
        as_dataframe: bool = False,
    ):
        if self.buffer_type == BufferType.DataFrame:
            slice = self.buffer.iloc[start:end]
        else:
            slice = self.buffer[start:end]
        if as_dataframe and not isinstance(slice, pd.DataFrame):
            return pd.DataFrame(slice)
        else:
            return slice

    def _write_buffer(self, force=False):
        if self.buffer is None:
            return
        while len(self.buffer) >= self.buffer_size:
            self.writer.append_data(
                self._buffer_slice(end=self.buffer_size, as_dataframe=True)
            )
            self.buffer = self._buffer_slice(
                start=self.buffer_size,
            )
        if force and len(self.buffer) > 0:
            slice = self._buffer_slice(as_dataframe=True)
            self.writer.append_data(slice)
            self.buffer = None

    def append_data(self, data: pd.DataFrame | dict | list[dict] | np.record):
        assert self.initialized and not self.finalized

        if self.buffer_type == BufferType.DataFrame:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    f"Parameter `data` must be of type DataFrame, not {type(data)}"
                )

            if self.buffer is None:
                self.buffer = data.copy(deep=True)
            else:
                self.buffer = pd.concat([self.buffer, data], axis=0, ignore_index=True)
        elif self.buffer_type == BufferType.Dicts:
            if isinstance(data, dict):
                data = [data]
            if not (isinstance(data, list) and isinstance(data[0], dict)):
                raise TypeError(
                    "Parameter `data` must be of type dict or list[dict],"
                    f" not {type(data)}"
                )
            if self.buffer is None:
                self.buffer = []
            self.buffer += data
        elif self.buffer_type == BufferType.Records:
            if self.buffer is None:
                self.buffer = np.recarray(shape=(0,), dtype=data.dtype)
            self.buffer = np.append(self.buffer, data)
        else:
            raise ValueError(f"Unknown buffer type {self.buffer_type}")

        self._write_buffer()

    def check_valid_data(self, data: pd.DataFrame):
        return self.writer.check_valid_data(data)

    def write(self, data: pd.DataFrame):
        self.writer.write(data)

    def initialize(self):
        assert not self.initialized
        self.initialized = True
        self.writer.initialize()

    def finalize(self):
        assert self.initialized
        self.finalized = True  # Only for checking whether this got called
        self._write_buffer(force=True)
        self.writer.finalize()

    def get_associated_reader(self):
        return self.writer.get_associated_reader()
