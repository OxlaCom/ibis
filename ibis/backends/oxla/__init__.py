"""Oxla backend."""

from __future__ import annotations

import contextlib
import inspect
import warnings
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import psycopg
import sqlglot as sg
import sqlglot.expressions as sge
from pandas.api.types import is_float_dtype

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, CanListCatalog, PyArrowExampleLoader
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import TRUE, C, ColGen

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self


class NatDumper(psycopg.adapt.Dumper):
    def dump(self, obj, context: Any | None = None) -> str | None:
        return None


class Backend(SQLBackend, CanCreateDatabase, CanListCatalog, PyArrowExampleLoader):
    name = "oxla"
    compiler = sc.oxla.compiler
    supports_python_udfs = True

    def _from_url(self, url: ParseResult, **kwargs):
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments

        Returns
        -------
        BaseBackend
            A backend instance

        """
        database, *schema = url.path[1:].split("/", 1)
        connect_args = {
            "user": url.username,
            "password": unquote_plus(url.password or ""),
            "host": url.hostname,
            "port": url.port,
        }

        kwargs.update(connect_args)
        self._convert_kwargs(kwargs)

        if "user" in kwargs and not kwargs["user"]:
            del kwargs["user"]

        if "host" in kwargs and not kwargs["host"]:
            del kwargs["host"]

        if "password" in kwargs and kwargs["password"] is None:
            del kwargs["password"]

        if "port" in kwargs and kwargs["port"] is None:
            del kwargs["port"]

        return self.connect(**kwargs)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """No-op."""

    def _finalize_memtable(self, name: str) -> None:
        """No-op."""

    def _fetch_from_cursor(
        self, cursor: psycopg.Cursor, schema: sch.Schema
    ) -> pd.DataFrame:
        import pandas as pd

        from ibis.formats.pandas import PandasData

        try:
            df = pd.DataFrame.from_records(
                cursor.fetchall(), columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            #
            # in the sqlite case failing to close the cursor results in
            # artificially locked tables
            cursor.close()
            raise
        df = PandasData.convert_table(df, schema)
        return df

    @property
    def version(self):
        version = f"{self.con.info.server_version:0>6}"
        major = int(version[:2])
        minor = int(version[2:4])
        patch = int(version[4:])
        pieces = [major]
        if minor:
            pieces.append(minor)
        pieces.append(patch)
        return ".".join(map(str, pieces))

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to Oxla database.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port number
        kwargs
            Additional keyword arguments to pass to the backend client connection.
        """
        import pandas as pd
        import psycopg
        import psycopg.types.json

        psycopg.types.json.set_json_loads(loads=lambda x: x)

        self.con = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            autocommit=True,
            **kwargs,
        )

        self.con.adapters.register_dumper(type(pd.NaT), NatDumper)

    @util.experimental
    @classmethod
    def from_connection(cls, con: psycopg.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to an Oxla database.

        Parameters
        ----------
        con
            An existing connection to an Oxla database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        return new_backend

    def list_tables(
        self, *, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """List the tables in `database` matching the pattern `like`.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Database to list tables from. Default behavior is to show tables in
            the current database.

        Returns
        -------
        list[str]
            List of table names in `database` matching the optional pattern
            `like`.
        """

        if database is not None:
            table_loc = database
        else:
            table_loc = (self.current_catalog, self.current_database)

        table_loc = self._to_sqlglot_table(table_loc)

        conditions = [TRUE]

        if (db := table_loc.args["db"]) is not None:
            db.args["quoted"] = False
            db = db.sql(dialect=self.dialect)
            conditions.append(C.table_schema.eq(sge.convert(db)))
        if (catalog := table_loc.args["catalog"]) is not None:
            catalog.args["quoted"] = False
            catalog = catalog.sql(dialect=self.dialect)
            conditions.append(C.table_catalog.eq(sge.convert(catalog)))

        sql = (
            sg.select("table_name")
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql(self.dialect)
        )

        con = self.con
        with con.cursor() as cursor, con.transaction():
            out = cursor.execute(sql).fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        cats = (
            sg.select(C.datname)
            .from_(sg.table("pg_database", db="pg_catalog"))
            .where(sg.not_(C.datistemplate))
            .sql(self.dialect)
        )
        con = self.con
        with con.cursor() as cursor, con.transaction():
            catalogs = list(map(itemgetter(0), cursor.execute(cats)))

        return self._filter_with_like(catalogs, like)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        dbs = (
            sg.select(C.nspname)
            .from_(sg.table("pg_namespace", db="pg_catalog"))
            .sql(self.dialect)
        )
        con = self.con
        with con.cursor() as cursor, con.transaction():
            databases = list(map(itemgetter(0), cursor.execute(dbs)))

        return self._filter_with_like(databases, like)

    @property
    def current_catalog(self) -> str:
        sql = sg.select(sg.func("current_database")).sql(self.dialect)
        con = self.con
        with con.cursor() as cursor, con.transaction():
            (db,) = cursor.execute(sql).fetchone()
        return db

    @property
    def current_database(self) -> str:
        sql = sg.select(sg.func("current_schema")).sql(self.dialect)
        con = self.con
        with con.cursor() as cursor, con.transaction():
            (schema,) = cursor.execute(sql).fetchone()
        return schema

    def function(self, name: str, *, database: str | None = None) -> Callable:
        n = ColGen(table="n")
        p = ColGen(table="p")
        f = self.compiler.f

        predicates = [p.proname.eq(sge.convert(name))]

        if database is not None:
            predicates.append(n.nspname.rlike(sge.convert(f"^({database})$")))

        query = (
            sg.select(
                f["pg_catalog.pg_get_function_result"](p.oid).as_("return_type"),
                f.string_to_array(
                    f["pg_catalog.pg_get_function_arguments"](p.oid), ", "
                ).as_("signature"),
            )
            .from_(sg.table("pg_proc", db="pg_catalog").as_("p"))
            .join(
                sg.table("pg_namespace", db="pg_catalog").as_("n"),
                on=n.oid.eq(p.pronamespace),
                join_type="LEFT",
            )
            .where(*predicates)
            .sql(self.dialect)
        )

        def split_name_type(arg: str) -> tuple[str, dt.DataType]:
            name, typ = arg.split(" ", 1)
            return name, self.compiler.type_mapper.from_string(typ)

        con = self.con
        with con.cursor() as cursor, con.transaction():
            rows = cursor.execute(query).fetchall()

        if not rows:
            name = f"{database}.{name}" if database else name
            raise exc.MissingUDFError(name)
        elif len(rows) > 1:
            raise exc.AmbiguousUDFError(name)

        [(raw_return_type, signature)] = rows
        return_type = self.compiler.type_mapper.from_string(raw_return_type)
        signature = list(map(split_name_type, signature))

        # dummy callable
        def fake_func(*args, **kwargs): ...

        fake_func.__name__ = name
        fake_func.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typ
                )
                for name, typ in signature
            ],
            return_annotation=return_type,
        )
        fake_func.__annotations__ = {"return": return_type, **dict(signature)}
        op = ops.udf.scalar.builtin(fake_func, database=database)
        return op

    def get_schema(
        self,
        name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ):
        a = ColGen(table="a")
        c = ColGen(table="c")
        n = ColGen(table="n")

        format_type = self.compiler.f["pg_catalog.format_type"]

        # If no database is specified, assume the current database
        db = database or self.current_database

        dbs = [sge.convert(db)]

        type_info = (
            sg.select(
                a.attname.as_("column_name"),
                format_type(a.atttypid, a.atttypmod).as_("data_type"),
                sg.not_(a.attnotnull).as_("nullable"),
            )
            .from_(sg.table("pg_attribute", db="pg_catalog").as_("a"))
            .join(
                sg.table("pg_class", db="pg_catalog").as_("c"),
                on=c.oid.eq(a.attrelid),
                join_type="INNER",
            )
            .join(
                sg.table("pg_namespace", db="pg_catalog").as_("n"),
                on=n.oid.eq(c.relnamespace),
                join_type="INNER",
            )
            .where(
                a.attnum > 0,
                sg.not_(a.attisdropped),
                n.nspname.isin(*dbs),
                c.relname.eq(sge.convert(name)),
            )
            .order_by(a.attnum)
            .sql(self.dialect)
        )

        type_mapper = self.compiler.type_mapper

        con = self.con
        with con.cursor() as cursor, con.transaction():
            rows = cursor.execute(type_info).fetchall()

        if not rows:
            raise com.TableNotFound(name)

        return sch.Schema(
            {
                col: type_mapper.from_string(typestr, nullable=nullable)
                for col, typestr, nullable in rows
            }
        )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        raise com.UnsupportedOperationError(
            "Oxla does not support views"
        )

    def create_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        if catalog is not None and catalog != self.current_catalog:
            raise exc.UnsupportedOperationError(
                f"{self.name} does not support creating a database in a different catalog"
            )
        sql = sge.Create(
            kind="SCHEMA", this=sg.table(name, catalog=catalog), exists=force
        ).sql(self.dialect)
        con = self.con
        with con.cursor() as cursor, con.transaction():
            cursor.execute(sql)

    def drop_database(
        self,
        name: str,
        /,
        *,
        catalog: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        if catalog is not None and catalog != self.current_catalog:
            raise exc.UnsupportedOperationError(
                f"{self.name} does not support dropping a database in a different catalog"
            )

        sql = sge.Drop(
            kind="SCHEMA",
            this=sg.table(name, catalog=catalog),
            exists=force,
            cascade=cascade,
        ).sql(self.dialect)

        con = self.con
        with con.cursor() as cursor, con.transaction():
            cursor.execute(sql)

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        overwrite: bool = False,
    ):
        """Create a table in Oxla.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect

        table_expr = sg.table(temp_name, db=database, quoted=quoted)
        target = sge.Schema(this=table_expr, expressions=schema.to_sqlglot(dialect))

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        ).sql(dialect)

        this = sg.table(name, catalog=database, quoted=quoted)
        this_no_catalog = sg.table(name, quoted=quoted)

        con = self.con
        with con.cursor() as cursor, con.transaction():
            cursor.execute(create_stmt)

            if query is not None:
                insert_stmt = sge.Insert(this=table_expr, expression=query).sql(dialect)
                cursor.execute(insert_stmt)

            if overwrite:
                cursor.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect)
                )
                cursor.execute(
                    f"ALTER TABLE IF EXISTS {table_expr.sql(dialect)} RENAME TO {this_no_catalog.sql(dialect)}"
                )

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        drop_stmt = sg.exp.Drop(
            kind="TABLE",
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            exists=force,
        ).sql(self.dialect)
        con = self.con
        with con.cursor() as cursor, con.transaction():
            cursor.execute(drop_stmt)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str | sg.Expression, **kwargs: Any):
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        with (con := self.con).cursor() as cursor, con.transaction():
            yield cursor.execute(query, **kwargs)

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        con = self.con
        cursor = con.cursor()

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            cursor.close()
            raise
        else:
            return cursor

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        import pandas as pd
        import pyarrow as pa

        def _batches(self: Self, *, schema: pa.Schema, query: str):
            con = self.con
            columns = schema.names
            # server-side cursors need to be uniquely named
            with (
                con.cursor(name=util.gen_name("postgres_cursor")) as cursor,
                con.transaction(),
            ):
                cursor.execute(query)
                while batch := cursor.fetchmany(chunk_size):
                    yield pa.RecordBatch.from_pandas(
                        pd.DataFrame(batch, columns=columns), schema=schema
                    )

        self._run_pre_execute_hooks(expr)

        schema = expr.as_table().schema().to_pyarrow()
        query = self.compile(expr, limit=limit, params=params)
        return pa.RecordBatchReader.from_batches(
            schema, _batches(self, schema=schema, query=query)
        )
