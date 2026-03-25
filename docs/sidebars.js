/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    'introduction/overview',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/quick-start',
        'getting-started/admin-ui',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'core-concepts/model-and-api-key',
          label: 'Model and API Key',
        },
        'core-concepts/provider-abstraction',
        'core-concepts/dynamic-configuration',
        'core-concepts/request-lifecycle-hooks',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/model-management',
        'guides/authentication',
        'guides/rate-limiting',
        'guides/request-streaming',
      ],
    },
    {
      type: 'category',
      label: 'Providers',
      items: [],
    },
    {
      type: 'category',
      label: 'Features',
      items: [],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [],
    },
    {
      type: 'category',
      label: 'Observability',
      items: [],
    },
    {
      type: 'category',
      label: 'Integrations',
      items: [],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [],
    },
  ],
};

module.exports = sidebars;
