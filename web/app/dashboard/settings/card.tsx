import { Card, Divider, Text, Title, TextInput, Button, Group } from '@mantine/core';
import Models from './combobox';

const cardData = [
  {
    title: 'Email',
    content:
      'Permanently remove your Personal Account and all of its contents from Infrarail. This action is not reversible, so please continue with caution.',
    inputType: 'text',
    buttonText: 'Save',
    buttonColor: 'black',
  },
  {
    title: 'Models',
    content:
      'Permanently remove your Personal Account and all of its contents from Infrarail. This action is not reversible, so please continue with caution.',
    inputType: 'models',
    buttonText: 'Save',
    buttonColor: 'black',
  },
  {
    title: 'Delete Account',
    content:
      'Permanently remove your Personal Account and all of its contents from Infrarail. This action is not reversible, so please continue with caution.',
    inputType: 'text',
    buttonText: 'Delete Personal Account',
    buttonColor: 'red',
  },
];

export default function CardComponent() {
  return (
    <>
      {cardData.map((card, index) => (
        <Card key={index} padding="lg" radius="md" withBorder mb={20}>
          <Card.Section pl="xl" pt="md" withBorder>
            <Title order={4}>{card.title}</Title>
            <Text mt={12} fz="14px" fw={400}>
              {card.content}
            </Text>
            {card.inputType === 'text' && <TextInput w={320} mb={22} mt={10} />}
            {card.inputType === 'models' && <Models />}
          </Card.Section>

          <Group justify="space-between" mt="xs">
            <Text fw={500} c="dimmed">
              We will email you to verify the change.
            </Text>
            <Button color={card.buttonColor} radius="md">
              {card.buttonText}
            </Button>
          </Group>
        </Card>
      ))}
    </>
  );
}
