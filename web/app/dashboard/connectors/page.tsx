"use client";
import React from "react";
import { Text, SimpleGrid, Container, Button,Modal } from "@mantine/core";
import Link from "next/link";
import classes from "./FeaturesAsymmetrical.module.css";
import Image from 'next/image'
import {
  IconBrandNotion,
  IconBrandGoogleDrive,
  IconMail,
  IconBrandAirtable,
  IconJson,
  IconCsv,
  IconPdf,
  IconDatabase,
} from "@tabler/icons-react";
import { useDisclosure } from '@mantine/hooks';
import { modals } from '@mantine/modals';
import { redirect } from "next/dist/server/api-utils";
const integrationsData = [
  {
    id: 1,
    title: "Notion",
    description: "Integrate with Notion to manage your tasks.",
    slug: "https://api.notion.com/v1/oauth/authorize?owner=user&client_id=f68bed3b-fa26-43a0-a603-4a1db2f2b3b7&redirect_uri=http://localhost:3000/dashboard/connectors/callback&response_type=code",
    icon: <Image src="/notion.png" alt="me" width="30" height="30" />,
  },
  {
    id: 2,
    title: "Google Drive",
    description: "Access and manage your Google Drive files.",
    slug: "/integration-google-drive",
    icon: <IconBrandGoogleDrive />,
  },
  {
    id: 3,
    title: "Gmail",
    description: "Connect with Gmail to handle your emails.",
    slug: "/integration-gmail",
    icon: <IconMail />,
  },
  {
    id: 4,
    title: "Confluence",
    description: "Integrate with Confluence for documentation.",
    slug: "/integration-confluence",
    icon: <IconBrandAirtable />,
  },
  {
    id: 5,
    title: "Database",
    description: "Integrate with Confluence for documentation.",
    slug: "/integration-confluence",
    icon: <IconDatabase />,
  },
  {
    id: 6,
    title: "Json",
    description: "Integrate with Json for documentation.",
    slug: "/integration-confluence",
    icon: <IconJson />,
  },
  {
    id: 7,
    title: "Pdf",
    description: "Integrate with PDF for documentation.",
    slug: "/integration-confluence",
    icon: <IconPdf />,
  },
  // ... add more integrations as needed
];

const Integrations = () => {
  const openModal = (slug:string) => modals.openConfirmModal({
    title: 'Please confirm your action',
    centered: true,
    children: (
      <Text size="sm">
        This action is so important that you are required to confirm it with a modal. Please click
        one of these buttons to proceed.
      </Text>
    ),
    labels: { confirm: 'Confirm', cancel: 'Cancel' },
    confirmProps: { color: 'black' },
    onCancel: () => console.log('Cancel'),
    onConfirm: () =>   window.open(slug, '_blank')
  });
  return (
    <Container size="xl">
      <Text size="xl" style={{ marginBottom: "20px" }}>
        Available Integrations
      </Text>
      <div style={{ marginLeft: "-80px" }}>
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing={50}>
          {integrationsData.map((integration) => (
            <div key={integration.id} className={classes.boxP}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  marginBottom: "10px",
                  padding: "10px",
                }}
              >
                {integration.icon}
                <Text size="lg" style={{ marginLeft: "10px" }}>
                  {integration.title}
                </Text>
              </div>
              <Text
                size="sm"
                style={{ marginBottom: "10px", paddingLeft: "10px" }}
              >
                {integration.description}
              </Text>
          
              <Button
                onClick={() => openModal(integration.slug)}
                
                style={{
                  cursor: "pointer",
                  marginLeft: "110px",
                  marginTop: "30px",
                }}
                size="compact-md"
                color="black"
                variant="filled"
              >
                Connect
              </Button>
            </div>
          ))}
        </SimpleGrid>
      </div>
    </Container>
  );
};

export default Integrations;
