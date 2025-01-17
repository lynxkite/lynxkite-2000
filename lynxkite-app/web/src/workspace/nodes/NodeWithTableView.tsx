import { useState } from 'react';
import Markdown from 'react-markdown'
import LynxKiteNode from './LynxKiteNode';
import Table from './Table';
import React from 'react';

function toMD(v: any): string {
  if (typeof v === 'string') {
    return v;
  }
  if (Array.isArray(v)) {
    return v.map(toMD).join('\n\n');
  }
  return JSON.stringify(v);
}

export default function NodeWithTableView(props: any) {
  const [open, setOpen] = useState({} as { [name: string]: boolean });
  const display = props.data.display?.value;
  const single = display?.dataframes && Object.keys(display?.dataframes).length === 1;
  return (
    <LynxKiteNode {...props}>
      {display && [
        Object.entries(display.dataframes || {}).map(([name, df]: [string, any]) => <React.Fragment key={name}>
          {!single && <div key={name + '-header'} className="df-head" onClick={() => setOpen({ ...open, [name]: !open[name] })}>{name}</div>}
          {(single || open[name]) &&
            (df.data.length > 1 ?
              <Table key={name + '-table'} columns={df.columns} data={df.data} />
              :
              <dl key={name + '-dl'}>
                {df.columns.map((c: string, i: number) =>
                  <React.Fragment key={name + '-' + c}>
                    <dt>{c}</dt>
                    <dd><Markdown>{toMD(df.data[0][i])}</Markdown></dd>
                  </React.Fragment>)
                }
              </dl>)}
        </React.Fragment>),
        Object.entries(display.others || {}).map(([name, o]) => <>
          <div className="df-head" onClick={() => setOpen({ ...open, [name]: !open[name] })}>{name}</div>
          {open[name] && <pre>{(o as any).toString()}</pre>}
        </>
        )]}
    </LynxKiteNode >
  );
}
