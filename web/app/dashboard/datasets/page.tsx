"use client";
import React from "react";
import { Text, SimpleGrid, Container, Button,Modal,TextInput,Group,rem,Title } from "@mantine/core";
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


const Datasource = () => {
  const [opened, { open, close }] = useDisclosure(false);

  return (
      <>
      <Text>You have not created any data source.</Text>
      <Modal opened={opened} onClose={close} withCloseButton={false} centered size="55%" radius={20} padding={24}>
                <Title order={3} mb={24}>Import your website</Title>
                <Text mt="5">
                  We’ll crawl your site and pull in as much of your content as possible.
                </Text>
                <Group mt={10}>
                
                  <Button >
                    Add
                  </Button>
                </Group>
                <Group mt={10}>

                  <TextInput
                    label="Sitemap"
                    withAsterisk

                    placeholder="https://example.com/sitemap.xml"
                  />
                  <Button variant="filled" mt="30" color="black" size="xs" loading>Get links</Button>
                </Group>
                <Text mt="20">Found 755 links</Text>
            
                <Button mt="10" ml={590} radius={6}>Save</Button>
              </Modal>
      <Button onClick={open}>Add a dataset</Button>
      
      </>
  );
};

export default Datasource;
