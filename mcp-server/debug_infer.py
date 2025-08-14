#!/usr/bin/env python3
import asyncio
import traceback
from pathlib import Path

from eda_agent import load_data, infer_schema, LoadDataRequest, SchemaInferenceRequest


async def main():
    try:
        data_path = str(Path(__file__).parent / 'iris.csv')
        await load_data(LoadDataRequest(path=data_path, name='iris_dataset', file_type='csv'))
        res = await infer_schema(SchemaInferenceRequest(name='iris_dataset'))
        print('OK')
        print(res.model_dump_json(indent=2))
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())


