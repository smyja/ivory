'use client'
import { Container, Grid, Card, Text, rem, Divider, Button, Group, Switch, useMantineTheme, Badge } from '@mantine/core';
import ReactMarkdown from 'react-markdown';
import { IconCopy, IconCheck, IconX } from '@tabler/icons-react';
import { useState } from 'react';
import { AccordionDemo } from '../accordion'; // Adjust the import path as needed

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <Button 
      onClick={handleCopy} 
      size="sm" 
      variant="subtle" 
      color={copied ? 'teal' : 'gray'}
      px={0}
    >
      {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
    </Button>
  );
}

export default function LeadGrid() {
  const theme = useMantineTheme();
  const [renderMarkdown, setRenderMarkdown] = useState(false);

  const dummyInput = `
### Input Code

\`\`\`javascript
function greet(name) {
  return "Hello, " + name + "!";
}
\`\`\`
  `;

  const dummyOutput = `
# Output Code

\`\`\`javascript
function greet(name) {
  return \`Hello, \${name}!\`;
}
\`\`\`

The refactored code uses ES6 template literals (backticks) instead of string concatenation. 
This makes the code more readable and allows for easier insertion of variables into strings.
  `;

  const preStyles = {
    fontFamily: 'inherit',
    fontWeight: 300,
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
    width: '100%',
    margin: 0,
    padding: theme.spacing.sm,
    backgroundColor: 'aliceblue',
    borderRadius: theme.radius.md,
    border: '1px solid black',
  };

  return (
    <Container my="md" fluid>
      <Grid gutter="md">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <Card shadow="sm" radius="md" withBorder>
            <Card.Section withBorder inheritPadding py="xs" style={{backgroundColor:"lavenderblush"}}>
              <Group justify="space-between">
                <Text fw={300} size="sm">Code Refactoring Assistant</Text>
                <Switch
                  checked={renderMarkdown}
                  onChange={(event) => setRenderMarkdown(event.currentTarget.checked)}
                  color="teal"
                  size="sm"
                  label="Markdown"
                  thumbIcon={
                    renderMarkdown ? (
                      <IconCheck
                        style={{ width: rem(12), height: rem(12) }}
                        color={theme.colors.teal[6]}
                        stroke={3}
                      />
                    ) : (
                      <IconX
                        style={{ width: rem(12), height: rem(12) }}
                        color={theme.colors.red[6]}
                        stroke={3}
                      />
                    )
                  }
                />
              </Group>
            </Card.Section>

            <Card.Section inheritPadding mt="md">
              <Group justify="space-between" mb="xs">
                <Badge color="rgba(255, 110, 110, 1)">Prompt</Badge>
                <CopyButton text={dummyInput} />
              </Group>
              {renderMarkdown ? (
                <ReactMarkdown>{dummyInput}</ReactMarkdown>
              ) : (
                <pre style={preStyles}>{dummyInput}</pre>
              )}
            </Card.Section>

            <Divider my="md" />

            <Card.Section inheritPadding pb="md">
              <Group justify="space-between" mb="xs">
                <Badge color="rgba(255, 110, 110, 1)">Prompt</Badge>
                <CopyButton text={dummyOutput} />
              </Group>
              {renderMarkdown ? (
                <ReactMarkdown>{dummyOutput}</ReactMarkdown>
              ) : (
                <pre style={preStyles}>{dummyOutput}</pre>
              )}
            </Card.Section>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <AccordionDemo />
        </Grid.Col>
      </Grid>
    </Container>
  );
}