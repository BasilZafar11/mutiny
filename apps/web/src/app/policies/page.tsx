import Link from "next/link";
import { CodeBlock } from "@/components/ui/CodeBlock";

export default function PoliciesPage() {
  return (
    <div className="page page-narrow">
      <header className="page-header">
        <div>
          <h1 className="page-title">Policies</h1>
          <p className="page-sub">
            Edit policy.yaml in your local project. Hosted observes the results
            of local campaigns.
          </p>
        </div>
        <Link href="/campaigns" className="btn btn-primary">
          Observe campaigns
        </Link>
      </header>

      <section className="mt-8" aria-labelledby="local-policies">
        <h2 id="local-policies" className="text-lg font-semibold text-text">
          Policies are managed locally
        </h2>
        <p className="mt-3 text-sm text-muted">
          Hosted does not read or edit policy files on your machine. There is
          no policy editor here. Initialize your project with <code>mutiny init</code>,
          edit <code>policy.yaml</code>, then run a campaign from that project:
        </p>
        <CodeBlock copyable label="Run from your local project" className="mt-4">
          {"mutiny run --hosted"}
        </CodeBlock>
        <p className="mt-4 text-sm text-muted">
          The agent runs locally. Redacted evidence and policy metadata are
          synced to Hosted when the run completes. Run <code>mutiny test</code>
          {" "}locally to replay saved regressions.
        </p>
        <p className="mt-4 text-sm text-muted">
          The bundled demo uses its own sample policy. It does not load your
          project&apos;s policy.yaml.
        </p>
        <Link href="/projects" className="btn btn-secondary mt-6">
          View projects
        </Link>
      </section>
    </div>
  );
}
