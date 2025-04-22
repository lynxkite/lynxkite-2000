import { useReactFlow } from "@xyflow/react";
import { useState } from "react";
import React from "react";
import Markdown from "react-markdown";
// @ts-ignore
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

  function displayRelation(relation: any) {
    // TODO: Dynamic autocomplete
    return (
      <form
        className="graph-relation-attributes"
        onSubmit={(e) => {
          updateRelation(e, relation);
        }}
      >
        <label htmlFor="name">Name:</label>
        <input type="text" id="name" name="name" defaultValue={relation.name} />

        <label htmlFor="df">DataFrame:</label>
        <input
          type="text"
          id="df"
          name="df"
          defaultValue={relation.df}
          list="df-options"
          required
        />

        <label htmlFor="source_column">Source Column:</label>
        <input
          type="text"
          id="source_column"
          name="source_column"
          defaultValue={relation.source_column}
          list="edges-column-options"
          required
        />

        <label htmlFor="target_column">Target Column:</label>
        <input
          type="text"
          id="target_column"
          name="target_column"
          defaultValue={relation.target_column}
          list="edges-column-options"
          required
        />

        <label htmlFor="source_table">Source Table:</label>
        <input
          type="text"
          id="source_table"
          name="source_table"
          defaultValue={relation.source_table}
          list="df-options"
          required
        />

        <label htmlFor="target_table">Target Table:</label>
        <input
          type="text"
          id="target_table"
          name="target_table"
          defaultValue={relation.target_table}
          list="df-options"
          required
        />

        <label htmlFor="source_key">Source Key:</label>
        <input
          type="text"
          id="source_key"
          name="source_key"
          defaultValue={relation.source_key}
          list="source-node-column-options"
          required
        />

        <label htmlFor="target_key">Target Key:</label>
        <input
          type="text"
          id="target_key"
          name="target_key"
          defaultValue={relation.target_key}
          list="target-node-column-options"
          required
        />

        <datalist id="df-options">
          {Object.keys(tables).map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>

        <datalist id="edges-column-options">
          {tables[relation.source_table] &&
            tables[relation.df].columns.map((name: string) => <option key={name} value={name} />)}
        </datalist>

        <datalist id="source-node-column-options">
          {tables[relation.source_table] &&
            tables[relation.source_table].columns.map((name: string) => (
              <option key={name} value={name} />
            ))}
        </datalist>

        <datalist id="target-node-column-options">
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
              {(singleRelation || open[name]) && displayRelation(relation)}
            </React.Fragment>
          ))}
      </div>
    </div>
  );
}

export default LynxKiteNode(NodeWithGraphCreationView);
