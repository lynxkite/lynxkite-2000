import { useState } from "react";

function Cell({ value }: { value: any }) {
  if (typeof value === "string") {
    if (value.startsWith("https://") || value.startsWith("data:")) {
      return <img className="image-in-table" src={value} alt={value} />;
    }
    if (value.startsWith("<svg")) {
      // A data URL is safer than just dropping it in the DOM.
      const data = `data:image/svg+xml;base64,${btoa(value)}`;
      return <img className="image-in-table" src={data} alt={value} />;
    }
    return <>{value}</>;
  }
  return <>{JSON.stringify(value)}</>;
}

export default function Table(props: any) {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  function onClick(column: string) {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  }
  let data = props.data;
  const sortColumnIndex = props.columns.indexOf(sortColumn);
  if (sortColumnIndex >= 0) {
    data = [...props.data];
    data.sort((a: any[], b: any[]) => {
      const aValue = a[sortColumnIndex];
      const bValue = b[sortColumnIndex];
      if (aValue < bValue) {
        return sortDirection === "asc" ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortDirection === "asc" ? 1 : -1;
      }
      return 0;
    });
  }
  return (
    <table className="table-viewer">
      <thead>
        <tr>
          {props.columns.map((column: string) => (
            <th key={column} onClick={() => onClick(column)}>
              {column}
              {sortColumn === column && (
                <span className="sort-indicator">{sortDirection === "asc" ? "▲" : "▼"}</span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row: { [column: string]: any }, i: number) => (
          <tr key={`row-${i}`}>
            {props.columns.map((_column: string, j: number) => (
              <td key={`cell ${i}, ${j}`}>
                <Cell value={row[j]} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
