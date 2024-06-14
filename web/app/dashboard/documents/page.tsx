
"use client";
import React, { useState, ChangeEvent } from "react";
import {
    Avatar,
    ActionIcon,
    TextInput,
    Button,
    Paper,
    Text,
    Group,
    Center,
    Divider,
    List,
    ThemeIcon,
    rem,
    Title,
} from "@mantine/core";
import {
    IconSearch,
    IconArrowRight,
    IconCircleCheck,
    IconCircleDashed,
} from "@tabler/icons-react";

import axios from "axios";
import { DocumentTableSort } from "./Table";
export default function Documents() {
    return (
        <div>
            <Title>Documents</Title>
            <Text>       
                All files of the dataset are shown here, and the entire dataset can be linked to Dify citations or indexed via the Chat plugin.
            </Text>
            <DocumentTableSort/>
        </div>
    )
}


