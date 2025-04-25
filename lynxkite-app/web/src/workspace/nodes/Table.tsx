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
  return (
    <table className="table-viewer">
      <thead>
        <tr>
          {props.columns.map((column: string) => (
            <th key={column}>{column}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {props.data.map((row: { [column: string]: any }, i: number) => (
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
