"""
SYNTAX is used in ASTCalculator to calculate the AST similarity score.
OPERATOR are used in Tokenlization
KEYWORDS are used in WeightedBleuCalculator, should be words without spaces

"""


SQL_SYNTAX = {
        # basic SQL syntax
        'select ', 'from ', 'where ', 'group by ', 'order by ', 'having ',
        'join ', 'inner join', 'left join', 'right join', 'full join',
        'insert into', 'update ', 'delete from', 'create table', 'alter table',
        'drop table', 'truncate', 'union ', 'distinct', 'count(*)', 'as ',
        'between', 'like ', 'in (', 'exists', 'desc', 'asc',
        
        # data warehouse/ETL related SQL
        'partition by', 'over (', 'window', 'rank()', 'dense_rank()',
        'row_number()', 'lead(', 'lag(', 'first_value(', 'last_value(',
        'merge into', 'when matched', 'when not matched',
        'with ', 'cte', 'recursive', 'pivot', 'unpivot',
        
        # advanced functions
        'case when', 'coalesce(', 'nullif(', 'cast(', 'extract(',
        'date_trunc', 'to_char(', 'to_date(', 'regexp_replace(',
        'regexp_extract(', 'substring(', 'concat(', 'lower(', 'upper(',
    }

SQL_OPERATORS = {
    '<>', '<=', '>=', '||'
    }

PYTHON_SYNTAX = {            
    # basic python syntax
            'def ', 'import ', 'class ', 'if ', 'for ', 'while ', 'print(',
            'return ', 'with ', 'try:', 'except:', 'lambda ',
            'yield ', 'async ', 'await ', '__init__', '.py', 'self.', 'lower(', 'upper(',
            
            # data analysis related
            'pandas', 'pd.', 'np.', 'numpy', 'matplotlib', 'plt.',
            'df.', '.iloc', '.loc', '.groupby', '.agg', '.apply',
            '.drop', '.fillna', '.value_counts', '.describe', '.head',
            '.merge', '.join', '.concat', '.read_csv', '.to_csv',
            
            # PySpark related
            'spark.', 'sparkContext', 'SparkSession', 'SparkConf',
            'pyspark.', 'rdd.', '.rdd', '.collect()', '.count()',
            '.createOrReplaceTempView', '.registerTempTable',
            '.udf', '.withColumn', '.select', '.filter', '.where',
            '.withColumnRenamed', '.join', '.groupBy', '.agg',
            '.sql', '.createDataFrame', '.read.parquet', '.write.parquet',
            '.read.csv', '.write.csv', '.saveAsTextFile'}

PYTHON_OPERATORS = {
    '+=', '-=', '*=', '/=', '**', '//', '==', '!=', '>=', '<='
}


KEYWORDS = {
        # SQL keywords
        'select', 'from', 'where', 'group', 'by', 'having', 'order', 'join',
        'having', 'order', 'join',
        'inner', 'outer', 'left', 'right', 'on', 'union', 'all', 'insert',
        'update', 'delete', 'create', 'table', 'view', 'with', 'as', 'case',
        'when', 'then', 'else', 'end', 'distinct', 'between', 'like', 'in',
        'exists', 'not', 'null', 'and', 'or', 'count', 'sum', 'avg', 'min', 'max',
            
        # Python data transformation keywords
        'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'try',
        'except', 'import', 'from', 'class', 'lambda', 'map', 'filter', 'reduce',
        'zip', 'sorted', 'list', 'dict', 'set', 'tuple', 'append', 'extend',
        'pandas', 'numpy', 'pd', 'np', 'dataframe', 'series', 'array', 'groupby',
        'agg', 'apply', 'transform', 'pivot', 'merge', 'join', 'concat', 'value_counts',
        
        # PySpark keywords
        'spark', 'dataframe', 'rdd', 'select', 'where', 'filter', 'map',
        'flatmap', 'groupby', 'agg', 'withcolumn', 'join', 'unionall',

        # C# keywords LINQ operations
        'where', 'select', 'first', 'firstordefault', 'single', 'singleordefault',
        'orderby', 'orderbydescending', 'groupby', 'join', 'count', 'sum', 'average',
        'min', 'max', 'any', 'all', 'contains', 'distinct', 'skip', 'take', 'toarray',
        'todictionary', 'aggregate'
        }

CSHARP_SYNTAX = {
    # basic C# syntax
    'using ', 'namespace ', 'class ', 'interface ', 'struct ', 'enum ',
    'public ', 'private ', 'protected ', 'internal ', 'static ', 'void ',
    'return ', 'if ', 'else ', 'for ', 'foreach ', 'while ', 'do ',
    'switch ', 'case ', 'break ', 'continue ', 'try ', 'catch ', 'finally ',
    'throw ', 'new ', 'this ', 'base ', 'async ', 'await ', 'var ', 'const ',
    'readonly ', 'get;', 'set;', '=>', 'delegate ', 'event ', 'lock ',
    
    # LINQ operations
    '.Where(', '.Select(', '.First(', '.FirstOrDefault(', '.Single(', '.SingleOrDefault(',
    '.OrderBy(', '.OrderByDescending(', '.GroupBy(', '.Join(', '.Count(', '.Sum(',
    '.Average(', '.Min(', '.Max(', '.Any(', '.All(', '.Contains(', '.Distinct(',
    '.Skip(', '.Take(', '.ToList(', '.ToArray(', '.ToDictionary(', '.Aggregate(',
    
    # Entity Framework/Database operations
    'DbContext', 'DbSet', '.Include(', '.ThenInclude(', '.AsNoTracking(',
    '.Where(', '.FirstOrDefault(', '.ToList(', '.SaveChanges(', '.Add(',
    '.Update(', '.Remove(', '.Find(', '.FromSqlRaw(', '.ExecuteSqlRaw(',
    
    # Common methods and properties
    '.ToString(', '.Equals(', '.GetHashCode(', '.CompareTo(', '.Substring(',
    '.Split(', '.Join(', '.Replace(', '.Trim(', '.ToLower(', '.ToUpper(',
    '.StartsWith(', '.EndsWith(', '.Contains(', '.IndexOf(', '.LastIndexOf(',
    '.Length', '.Count', '.Add(', '.Remove(', '.Clear(', '.Contains(',
}

CSHARP_OPERATORS = {
    '??', '?.', '??=', '=>', '++', '--', '+=', '-=', '*=', '/=', '%=',
    '&=', '|=', '^=', '<<=', '>>=', '==', '!=', '>=', '<=', '&&', '||',
    '<<', '>>', '&', '|', '^', '~', '!', '?:', '??', '?.', '??=', '=>'
}
