
BEGIN;

LOCK TABLE public.payment_attempts IN ACCESS EXCLUSIVE MODE;

DO $migration$
DECLARE
    existing_constraint_oid OID;
    existing_constraint_type "char";
    existing_definition TEXT;
    existing_column_count INTEGER;
    existing_columns TEXT[];
BEGIN
    SELECT
        c.oid,
        c.contype,
        pg_get_constraintdef(c.oid),
        cardinality(c.conkey),
        ARRAY(
            SELECT a.attname
            FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
            JOIN pg_attribute AS a
              ON a.attrelid = c.conrelid
             AND a.attnum = k.attnum
            ORDER BY k.ord
        )
    INTO
        existing_constraint_oid,
        existing_constraint_type,
        existing_definition,
        existing_column_count,
        existing_columns
    FROM pg_constraint AS c
    WHERE c.conrelid = 'public.payment_attempts'::regclass
      AND c.conname = 'uq_payment_attempt_transaction_number';

    IF existing_constraint_oid IS NOT NULL THEN
        IF existing_constraint_type <> 'u'
           OR existing_column_count <> 2
           OR existing_columns IS DISTINCT FROM
              ARRAY['transaction_id', 'attempt_number']::TEXT[] THEN
            RAISE EXCEPTION
                'Constraint exists with unexpected definition: %',
                existing_definition;
        END IF;

        RAISE NOTICE 'Expected uniqueness constraint already exists.';
    ELSE
        IF EXISTS (
            SELECT 1
            FROM public.payment_attempts
            GROUP BY transaction_id, attempt_number
            HAVING COUNT(*) > 1
        ) THEN
            RAISE EXCEPTION
                'Duplicate transaction_id and attempt_number pairs exist. Migration aborted.';
        END IF;

        ALTER TABLE public.payment_attempts
            ADD CONSTRAINT uq_payment_attempt_transaction_number
            UNIQUE (transaction_id, attempt_number);
    END IF;
END
$migration$;

COMMIT;
