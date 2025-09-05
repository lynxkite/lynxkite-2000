import { useReactFlow } from "@xyflow/react";
import React, { type FormEventHandler, useState } from "react";
import Markdown from "react-markdown";
// @ts-expect-error
import Trash from "~icons/tabler/trash";
import LynxKiteNode from "./LynxKiteNode";
import Table from "./Table";

function toMD(v: any): string {
  if (typeof v === "string") {
    return v;
  }
  if (Array.isArray(v)) {
    return v.map(toMD).join("\n\n");
  }
  return JSON.stringify(v);
}

function displayTable(name: string, df: any) {
  if (df.data.length > 1) {
    return (
      <Table key={`${name}-table`} name={`${name}-table`} columns={df.columns} data={df.data} />
    );
  }
  if (df.data.length) {
    return (
      <dl key={`${name}-dl`}>
        {df.columns.map((c: string, i: number) => (
          <React.Fragment key={`${name}-${c}`}>
            <dt>{c}</dt>
            <dd>
              <Markdown>{toMD(df.data[0][i])}</Markdown>
            </dd>
          </React.Fragment>
        ))}
      </dl>
    );
  }
  return JSON.stringify(df.data);
}

function relationsToDict(relations: any[]) {
  if (!relations) {
    return {};
  }
  return Object.assign({}, ...relations.map((r: any) => ({ [r.name]: r })));
}

export type UpdateOptions = { delay?: number };

function RelationView({
  relation,
  tables,
  onSubmit,
}: {
  relation: any;
  tables: { [name: string]: any };
  onSubmit: FormEventHandler<HTMLFormElement>;
}) {
  const idPrefix = React.useId();
  const ids = {
    name: `${idPrefix}-name`,
    df: `${idPrefix}-df`,
    source_column: `${idPrefix}-source_column`,
    target_column: `${idPrefix}-target_column`,
    source_table: `${idPrefix}-source_table`,
    target_table: `${idPrefix}-target_table`,
    source_key: `${idPrefix}-source_key`,
    target_key: `${idPrefix}-target_key`,
    df_options: `${idPrefix}-df-options`,
    edges_column_options: `${idPrefix}-edges-column-options`,
    source_node_column_options: `${idPrefix}-source-node-column-options`,
    target_node_column_options: `${idPrefix}-target-node-column-options`,
  };
  return (
    <form className="graph-relation-attributes" onSubmit={onSubmit}>
      <label htmlFor={ids.name}>Name:</label>
      <input type="text" id={ids.name} name="name" defaultValue={relation.name} />
      <label htmlFor={ids.df}>DataFrame:</label>
      <input
        type="text"
        id={ids.df}
        name="df"
        defaultValue={relation.df}
        list={ids.df_options}
        required
      />
      <label htmlFor={ids.source_column}>Source Column:</label>
      <input
        type="text"
        id={ids.source_column}
        name="source_column"
        defaultValue={relation.source_column}
        list={ids.edges_column_options}
        required
      />
      <label htmlFor={ids.target_column}>Target Column:</label>
      <input
        type="text"
        id={ids.target_column}
        name="target_column"
        defaultValue={relation.target_column}
        list={ids.edges_column_options}
        required
      />
      <label htmlFor={ids.source_table}>Source Table:</label>
      <input
        type="text"
        id={ids.source_table}
        name="source_table"
        defaultValue={relation.source_table}
        list={ids.df_options}
        required
      />
      <label htmlFor={ids.target_table}>Target Table:</label>
      <input
        type="text"
        id={ids.target_table}
        name="target_table"
        defaultValue={relation.target_table}
        list={ids.df_options}
        required
      />
      <label htmlFor={ids.source_key}>Source Key:</label>
      <input
        type="text"
        id={ids.source_key}
        name="source_key"
        defaultValue={relation.source_key}
        list={ids.source_node_column_options}
        required
      />
      <label htmlFor={ids.target_key}>Target Key:</label>
      <input
        type="text"
        id={ids.target_key}
        name="target_key"
        defaultValue={relation.target_key}
        list={ids.target_node_column_options}
        required
      />
      <datalist id={ids.df_options}>
        {Object.keys(tables).map((name) => (
          <option key={name} value={name} />
        ))}
      </datalist>
      <datalist id={ids.edges_column_options}>
        {tables[relation.source_table] &&
          tables[relation.df].columns.map((name: string) => <option key={name} value={name} />)}
      </datalist>
      <datalist id={ids.source_node_column_options}>
        {tables[relation.source_table] &&
          tables[relation.source_table].columns.map((name: string) => (
            <option key={name} value={name} />
          ))}
      </datalist>
      <datalist id={ids.target_node_column_options}>
        {tables[relation.source_table] &&
          tables[relation.target_table].columns.map((name: string) => (
            <option key={name} value={name} />
          ))}
      </datalist>
      <button className="submit-relationship-button" type="submit">
        Create
      </button>
    </form>
  );
}

function NodeWithGraphCreationView(props: any) {
  const reactFlow = useReactFlow();
  const [open, setOpen] = useState({} as { [name: string]: boolean });
  const display = props.data.display?.value;
  const tables = display?.dataframes || {};
  const singleTable = tables && Object.keys(tables).length === 1;
  const [relations, setRelations] = useState(relationsToDict(display?.relations) || {});
  const singleRelation = relations && Object.keys(relations).length === 1;
  function setParam(name: string, newValue: any, opts: UpdateOptions) {
    reactFlow.updateNodeData(props.id, {
      params: { ...props.data.params, [name]: newValue },
      __execution_delay: opts.delay || 0,
    });
  }

  function updateRelation(event: any, relation: any) {
    event.preventDefault();

    const updatedRelation = {
      ...relation,
      ...Object.fromEntries(new FormData(event.target).entries()),
    };

    // Avoid mutating React state directly
    const newRelations = { ...relations };
    if (relation.name !== updatedRelation.name) {
      delete newRelations[relation.name];
    }
    newRelations[updatedRelation.name] = updatedRelation;
    setRelations(newRelations);
    // There is some issue with how Yjs handles complex objects (maps, arrays)
    // so we need to serialize the relations object to a string
    setParam("relations", JSON.stringify(newRelations), {});
  }

  const addRelation = () => {
    const new_relation = {
      name: "new_relation",
      df: "",
      source_column: "",
      target_column: "",
      source_table: "",
      target_table: "",
      source_key: "",
      target_key: "",
    };
    setRelations({
      ...relations,
      [new_relation.name]: new_relation,
    });
    setOpen({ ...open, [new_relation.name]: true });
  };

  const deleteRelation = (relation: any) => {
    const newOpen = { ...open };
    delete newOpen[relation.name];
    setOpen(newOpen);
    const newRelations = { ...relations };
    delete newRelations[relation.name];
    setRelations(newRelations);
    // There is some issue with how Yjs handles complex objects (maps, arrays)
    // so we need to serialize the relations object to a string
    setParam("relations", JSON.stringify(newRelations), {});
  };

  return (
    <div className="graph-creation-view">
      <div className="graph-tables">
        <div className="graph-table-header">Node Tables</div>
        {display && [
          Object.entries(tables).map(([name, df]: [string, any]) => (
            <React.Fragment key={name}>
              {!singleTable && (
                <div
                  key={`${name}-header`}
                  className="df-head"
                  onClick={() => setOpen({ ...open, [name]: !open[name] })}
                >
                  {name}
                </div>
              )}
              {(singleTable || open[name]) && displayTable(name, df)}
            </React.Fragment>
          )),
          Object.entries(display.others || {}).map(([name, o]) => (
            <>
              <div
                key={name}
                className="df-head"
                onClick={() => setOpen({ ...open, [name]: !open[name] })}
              >
                {name}
              </div>
              {open[name] && <pre>{(o as any).toString()}</pre>}
            </>
          )),
        ]}
      </div>
      <div className="graph-relations">
        <div className="graph-table-header">
          Relationships
          <button className="add-relationship-button" onClick={(_) => addRelation()}>
            +
          </button>
        </div>
        {relations &&
          Object.entries(relations).map(([name, relation]: [string, any]) => (
            <React.Fragment key={name}>
              <div
                key={`${name}-header`}
                className="df-head"
                onClick={() => setOpen({ ...open, [name]: !open[name] })}
              >
                {name}
                <button
                  onClick={() => {
                    deleteRelation(relation);
                  }}
                >
                  <Trash />
                </button>
              </div>
              {(singleRelation || open[name]) && (
                <RelationView
                  relation={relation}
                  tables={tables}
                  onSubmit={(e) => updateRelation(e, relation)}
                />
              )}
            </React.Fragment>
          ))}
      </div>
    </div>
  );
}

export default LynxKiteNode(NodeWithGraphCreationView);
